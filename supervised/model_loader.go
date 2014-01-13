package supervised

import (
	"encoding/json"
	"log"
	"os"
)

func LoadModel(path string) Model {
	// 最大文件长度10M
	data := make([]byte, 1024*1024*10)

	f, err := os.Open(path)
	defer f.Close()
	if err != nil {
		log.Fatal("无法打开", path, "文件")
	}

	count, errRead := f.Read(data)
	if errRead != nil {
		log.Fatal("无法读入", path, "文件")
	}

	m := new(MaxEntClassifier)
	errUnmarshal := json.Unmarshal(data[:count], m)
	if errUnmarshal != nil {
		log.Fatal("无法解析", path, "文件，错误", errUnmarshal)
	}

	return m
}
