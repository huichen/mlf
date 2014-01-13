package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/supervised"
	"io"
	"log"
	"net/http"
	"runtime"
)

var (
	host  = flag.String("host", "127.0.0.1", "host名")
	port  = flag.Int("port", 8888, "端口")
	model = flag.String("model", "", "")

	classifier supervised.Model
)

type PredictResponse struct {
	Status       int
	ErrorMessage string
	Output       *data.InstanceOutput
}

func main() {
	flag.Parse()
	runtime.GOMAXPROCS(1)

	classifier = supervised.LoadModel(*model)

	http.HandleFunc("/predict", PredictRpc)
	log.Print("服务器启动 ", *host, ":", *port)
	http.ListenAndServe(fmt.Sprintf("%s:%d", *host, *port), nil)
}

func PredictRpc(w http.ResponseWriter, req *http.Request) {
	decoder := json.NewDecoder(req.Body)
	var instance data.Instance
	err := decoder.Decode(&instance)

	response := PredictResponse{}
	if err != nil {
		response.Status = 1
		response.ErrorMessage = fmt.Sprint(err)
	} else {
		response.Status = 0
	}

	output := classifier.Predict(&instance)
	response.Output = &output
	log.Print(output)

	w.Header().Set("Content-Type", "application/json")
	responseJson, _ := json.Marshal(&response)
	io.WriteString(w, string(responseJson))
}
