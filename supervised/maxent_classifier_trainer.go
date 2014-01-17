package supervised

import (
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/optimizer"
	"github.com/huichen/mlf/util"
	"log"
	"math"
)

// 最大熵分类训练器
type MaxEntClassifierTrainer struct {
	options TrainerOptions
}

// 创建一个最大熵分类训练器
func NewMaxEntClassifierTrainer(options TrainerOptions) Trainer {
	classifier := new(MaxEntClassifierTrainer)
	classifier.options = options
	return classifier
}

func (trainer *MaxEntClassifierTrainer) Train(set data.Dataset) Model {
	// 检查训练数据是否是分类问题
	if !set.GetOptions().IsSupervisedLearning {
		log.Fatal("训练数据不是分类问题数据")
	}

	// 建立新的优化器
	optimizer := optimizer.NewOptimizer(trainer.options.Optimizer)

	// 建立特征权重向量
	featureDimension := set.GetOptions().FeatureDimension
	numLabels := set.GetOptions().NumLabels
	var weights *util.Matrix
	if set.GetOptions().FeatureIsSparse {
		weights = util.NewSparseMatrix(numLabels-1)
	} else {
		weights = util.NewMatrix(numLabels-1, featureDimension)
	}

	// 得到优化的特征权重向量
	optimizer.OptimizeWeights(weights, MaxEntComputeInstanceDerivative, set)

	classifier := new(MaxEntClassifier)
	classifier.Weights = weights
	classifier.NumLabels = numLabels
	classifier.FeatureDimension = featureDimension
	classifier.FeatureDictionary = set.GetFeatureDictionary()
	classifier.LabelDictionary = set.GetLabelDictionary()
	return classifier
}

func MaxEntComputeInstanceDerivative(
	weights *util.Matrix, instance *data.Instance, instanceDerivative *util.Matrix) {
	// 定义偏导和特征向量
	features := instance.Features

	// 得到维度信息
	numLabels := weights.NumLabels() + 1

	// 计算 z = 1 + exp(sum(w_i * x_i))
	label := instance.Output.Label
	z := ComputeZ(weights, features, label, instanceDerivative)
	inverseZ := float64(1) / z

	for iLabel := 1; iLabel < numLabels; iLabel++ {
		vec := instanceDerivative.GetValues(iLabel - 1)
		if label == 0 || label != iLabel {
			vec.Multiply(inverseZ, 0, features)
		} else {
			vec.Multiply(inverseZ, -1, features)
		}
	}
}

// 计算 z = 1 + sum(exp(sum(w_i * x_i)))
//
// 在temp中保存 exp(sum(w_i * x_i))
func ComputeZ(weights *util.Matrix, features *util.Vector, label int, temp *util.Matrix) float64 {
	result := float64(1.0)
	numLabels := weights.NumLabels() + 1

	for iLabel := 1; iLabel < numLabels; iLabel++ {
		exp := math.Exp(util.VecDotProduct(features, weights.GetValues(iLabel-1)))
		result += exp

		tempVec := temp.GetValues(iLabel - 1)
		if tempVec.IsSparse() {
			for _, k := range features.Keys() {
				tempVec.Set(k, exp)
			}
		} else {
			tempVec.SetAll(exp)
		}
	}
	return result
}
