package rbm

import (
	"encoding/json"
	"github.com/huichen/mlf/util"
	"log"
	"os"
)

type RBMModel struct {
	Weights      *util.Matrix
	HiddenUnits  int
	VisibleUnits int
}

func (rbm *RBM) Write(path string) {
	rbm.lock.RLock()
	defer rbm.lock.RUnlock()

	model := RBMModel{
		Weights:      rbm.lock.weights,
		HiddenUnits:  rbm.lock.weights.NumLabels() - 1,
		VisibleUnits: rbm.lock.weights.NumValues() - 1,
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
