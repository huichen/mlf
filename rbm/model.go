package rbm

import (
	"encoding/json"
	"github.com/huichen/mlf/util"
	"log"
	"os"
)

type RBMModel struct {
	Weights *util.Matrix
	Options RBMOptions
}

func (rbm *RBM) Write(path string) {
	rbm.lock.RLock()
	defer rbm.lock.RUnlock()

	model := RBMModel{
		Weights: rbm.lock.weights,
		Options: rbm.options,
	}

	response, errMarshal := json.MarshalIndent(model, "", "\t")
	if errMarshal != nil {
		log.Fatal("无法转化模型为JSON格式")
	}
	f, err := os.Create(path)
	defer f.Close()
	if err != nil {
		log.Fatal("无法写入", path, "文件")
	}
	f.Write(response)
}

func LoadRBM(path string) *RBM {
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

	m := new(RBMModel)
	errUnmarshal := json.Unmarshal(data[:count], m)
	if errUnmarshal != nil {
		log.Fatal("无法解析", path, "文件，错误", errUnmarshal)
	}

	machine := &RBM{}
	machine.lock.weights = m.Weights
	machine.options = m.Options

	return machine
}
