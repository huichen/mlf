package main

import (
	"flag"
	"fmt"
	"github.com/petar/GoMNIST"
	"log"
)

var (
	data = flag.String("data", "", "MNIST文件目录")
)

func main() {
	flag.Parse()
	set, _, err := GoMNIST.Load(*data)
	//_, set, err := GoMNIST.Load(*data)
	if err != nil {
		log.Fatal("无法载入数据")
	}
	log.Printf("#images = %d", len(set.Images))

	for i := 0; i < len(set.Images); i++ {
		content := fmt.Sprintf("%d 1:0", set.Labels[i])
		image := set.Images[i]
		for index, p := range image {
			if p != 0 {
				content = fmt.Sprintf("%s %d:%0.3f", content, index+1, float32(p)/255.)
			}
		}
		fmt.Printf("%s %d:0\n", content, len(image))
		if i%1000 == 0 {
			log.Printf("已处理 %d 条记录", i)
		}
	}
}
