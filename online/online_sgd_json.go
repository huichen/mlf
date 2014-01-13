package online

import (
	"encoding/json"
	"github.com/huichen/mlf/supervised"
	"log"
	"os"
)

func (classifier *OnlineSGDClassifier) Write(path string) {
	model := supervised.MaxEntClassifier{}
	model.Weights = classifier.weights
	model.NumLabels = classifier.weights.NumLabels() + 1
	model.FeatureDictionary = classifier.featureDictionary
	model.LabelDictionary = classifier.labelDictionary

	response, errMarshal := json.MarshalIndent(model, "", "\t")
	if errMarshal != nil {
		log.Print("无法转化模型为JSON格式, err=", errMarshal)
	}
	f, err := os.Create(path)
	defer f.Close()
	if err != nil {
		log.Fatal("无法写入", path, "文件")
	}
	f.Write(response)
}

func (classifier *OnlineSGDClassifier) LoadWeightsFromFile(path string) {
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

	model := new(supervised.MaxEntClassifier)
	errUnmarshal := json.Unmarshal(data[:count], model)
	if errUnmarshal != nil {
		log.Fatal("无法解析", path, "文件，错误", errUnmarshal)
	}

	if classifier.options.NumLabels != model.NumLabels {
		log.Fatal("无法载入权重，标注数目不匹配")
	}
	classifier.weights = model.Weights
	classifier.featureDictionary = model.FeatureDictionary
	classifier.labelDictionary = model.LabelDictionary
}
