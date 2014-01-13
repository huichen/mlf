package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"github.com/huichen/mlf/contrib"
	"io"
	"io/ioutil"
	"log"
	"net/http"
)

var (
	input  = flag.String("input", "", "libsvm格式的数据文件，训练数据")
	server = flag.String("server", "localhost:8080", "SGD训练服务器地址")
	mode   = flag.String("mode", "train", "模式，train或者predict")
)

func main() {
	flag.Parse()

	set := contrib.LoadLibSVMDataset(*input, true)

	iterator := set.CreateIterator()

	client := &http.Client{}
	for {
		iterator.Start()
		for !iterator.End() {
			instance := iterator.GetInstance()

			if *mode != "train" {
				instance.Output = nil
			}

			httpBody, errMarshal := json.Marshal(instance)
			if errMarshal != nil {
				log.Print("无法JSON串行化样本")
			}
			req, _ := http.NewRequest("POST", "http://"+*server+"/train", bytes.NewReader(httpBody))
			req.Header.Set("Content-Type", "application/json")
			res, err := client.Do(req)
			io.Copy(ioutil.Discard, res.Body)
			if err != nil {
				log.Print("http请求失败, err=", err)
			} else {
				res.Body.Close()
			}

			iterator.Next()
		}
	}

}
