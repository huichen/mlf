package online

import (
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/dictionary"
	"github.com/huichen/mlf/eval"
	"github.com/huichen/mlf/optimizer"
	"github.com/huichen/mlf/supervised"
	"github.com/huichen/mlf/util"
)

// 在线梯度递降分类训练器
// 请使用NewOnlineSGDClassifier函数创建新的训练器
type OnlineSGDClassifier struct {
	weights            *util.Matrix
	derivative         *util.Matrix
	instanceDerivative *util.Matrix
	options            OnlineSGDClassifierOptions
	instancesProcessed int
	evaluator          OnlineEvaluator
	featureDictionary  *dictionary.Dictionary
	labelDictionary    *dictionary.Dictionary
}

// 从options中创建训练器
func NewOnlineSGDClassifier(options OnlineSGDClassifierOptions) *OnlineSGDClassifier {
	classifier := new(OnlineSGDClassifier)
	classifier.options = options
	if classifier.options.BatchSize <= 1 {
		classifier.options.BatchSize = 1
	}
	classifier.weights = util.NewSparseMatrix(options.NumLabels - 1)
	classifier.derivative = util.NewSparseMatrix(options.NumLabels - 1)
	classifier.instanceDerivative = util.NewSparseMatrix(options.NumLabels - 1)
	classifier.evaluator = new(FrapEvaluator)
	classifier.evaluator.Init(options.NumInstancesForEvaluation)
	classifier.featureDictionary = dictionary.NewDictionary(1)
	classifier.labelDictionary = dictionary.NewDictionary(0)

	return classifier
}

// 评价目前为止训练好的模型，得到评价metric
func (classifier *OnlineSGDClassifier) Evaluate() eval.Evaluation {
	return classifier.evaluator.Report()
}

// 读入一个训练样本
func (classifier *OnlineSGDClassifier) TrainOnOneInstance(instance *data.Instance) {
	if instance.NamedFeatures != nil {
		// 将样本中的特征转化为稀疏向量并加入词典
		instance.Features = nil
		data.ConvertNamedFeatures(instance, classifier.featureDictionary)
	}

	if instance.Output == nil {
		return
	} else {
		// 将样本中的标注字符串转化为整数ID
		if instance.Output.LabelString != "" {
			instance.Output.Label =
				classifier.labelDictionary.GetIdFromName(
					instance.Output.LabelString)
		}
	}

	// 预测并记录
	prediction := classifier.Predict(instance)
	classifier.evaluator.Evaluate(*instance.Output, prediction)

	classifier.instanceDerivative.Clear()
	supervised.MaxEntComputeInstanceDerivative(
		classifier.weights, instance, classifier.instanceDerivative)
	classifier.derivative.Increment(classifier.instanceDerivative, 1.0)
	classifier.instancesProcessed++

	if classifier.instancesProcessed >= classifier.options.BatchSize {
		// 添加正则化项
		classifier.derivative.Increment(optimizer.ComputeRegularization(
			classifier.weights, classifier.options.Optimizer), 1.0/float64(classifier.options.NumInstancesForEvaluation))

		// 根据学习率更新权重
		classifier.weights.Increment(
			classifier.derivative,
			-1*classifier.options.Optimizer.LearningRate/float64(classifier.options.NumInstancesForEvaluation))

		// 重置
		classifier.derivative.Clear()
		classifier.instancesProcessed = 0
	}
}

// 使用当前训练出的模型对一个样本的输出进行预测
func (classifier *OnlineSGDClassifier) Predict(instance *data.Instance) data.InstanceOutput {
	output := data.InstanceOutput{}

	predictedLabel := 0
	maxWeight := float64(0)
	for iLabel := 1; iLabel < classifier.weights.NumLabels()+1; iLabel++ {
		sum := float64(0)
		for _, k := range instance.Features.Keys() {
			sum += classifier.weights.Get(iLabel-1, k) * instance.Features.Get(k)
		}
		if sum > maxWeight {
			predictedLabel = iLabel
			maxWeight = sum
		}
	}
	output.Label = predictedLabel

	return output
}
