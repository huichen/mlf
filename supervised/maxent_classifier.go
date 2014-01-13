package supervised

import (
	"encoding/json"
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/dictionary"
	"github.com/huichen/mlf/util"
	"log"
	"math"
	"os"
)

// 最大熵分类器模型
type MaxEntClassifier struct {
	NumLabels        int
	FeatureDimension int

	LabelNames        []string
	FeatureDictionary *dictionary.Dictionary
	LabelDictionary   *dictionary.Dictionary

	Weights *util.Matrix
}

func (classifier *MaxEntClassifier) GetModelType() string {
	return "maxent_classifier"
}

func (classifier *MaxEntClassifier) Write(path string) {
	response, errMarshal := json.MarshalIndent(classifier, "", "\t")
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

func (classifier *MaxEntClassifier) Predict(instance *data.Instance) data.InstanceOutput {
	output := data.InstanceOutput{}

	// 当使用NamedFeatures时转化为Features
	if instance.NamedFeatures != nil {
		if classifier.FeatureDictionary == nil {
			return output
		}
		instance.Features = util.NewSparseVector()
		// 第0个feature始终是1
		instance.Features.Set(0, 1.0)

		for k, v := range instance.NamedFeatures {
			id := classifier.FeatureDictionary.TranslateIdFromName(k)
			instance.Features.Set(id, v)
		}
	}

	output.LabelDistribution = util.NewVector(classifier.NumLabels)
	output.LabelDistribution.Set(0, 1.0)

	z := float64(1)
	mostPossibleLabel := 0
	mostPossibleLabelWeight := float64(1)
	for iLabel := 1; iLabel < classifier.NumLabels; iLabel++ {
		sum := float64(0)
		for _, k := range classifier.Weights.GetValues(iLabel - 1).Keys() {
			sum += classifier.Weights.Get(iLabel-1, k) * instance.Features.Get(k)
		}
		exp := math.Exp(sum)
		if exp > mostPossibleLabelWeight {
			mostPossibleLabel = iLabel
			mostPossibleLabelWeight = exp
		}
		z += exp
		output.LabelDistribution.Set(iLabel, exp)
	}
	output.LabelDistribution.Scale(1 / z)
	output.Label = mostPossibleLabel

	if classifier.LabelDictionary != nil {
		output.LabelString = classifier.LabelDictionary.GetNameFromId(output.Label)
	}

	return output
}
