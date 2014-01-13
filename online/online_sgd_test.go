package online

import (
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/optimizer"
	"testing"
)

func TestOnlineSGD(t *testing.T) {
	options := OnlineSGDClassifierOptions{
		BatchSize:                 1,
		NumLabels:                 2,
		NumInstancesForEvaluation: 10,
		Optimizer: optimizer.OptimizerOptions{
			LearningRate:         0.01,
			RegularizationFactor: 0.0001,
			RegularizationScheme: 2,
		},
	}

	classifier := NewOnlineSGDClassifier(options)

	set := data.NewInmemDataset()
	set.AddInstance(&data.Instance{
		NamedFeatures: map[string]float64{
			"f1": 1,
			"f2": 4,
			"f3": 7,
		},
		Output: &data.InstanceOutput{
			Label: 0,
		},
	})
	set.AddInstance(&data.Instance{
		NamedFeatures: map[string]float64{
			"f1": 1,
			"f4": 4,
			"f5": 7,
			"f2": 7,
		},
		Output: &data.InstanceOutput{
			Label: 1,
		},
	})

	set.Finalize()
	iterator := set.CreateIterator()
	iterator.Start()
	for !iterator.End() {
		classifier.TrainOnOneInstance(iterator.GetInstance())
		iterator.Next()
	}

	classifier.Write("test.mlf")
}
