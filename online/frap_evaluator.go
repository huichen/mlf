package online

import (
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/eval"
	"github.com/huichen/mlf/util"
)

// F1、召回率（recall）、准确率（accuracy）、精度（precision）评价器
// 注意，前三个metric仅当问题是二分类问题时才有意义。如果是多分类，
// 返回的前三个metric为class-0-vs-rest意义下的值。
type FrapEvaluator struct {
	tp, tn, fp, fn, correct, total *util.CircularBuffer
}

// 初始化
// 评价最多最近size个样本的metric
func (e *FrapEvaluator) Init(size int) {
	e.tp = util.NewCircularBuffer(size)
	e.tn = util.NewCircularBuffer(size)
	e.fp = util.NewCircularBuffer(size)
	e.fn = util.NewCircularBuffer(size)
	e.correct = util.NewCircularBuffer(size)
}

// 添加一个评价样本，用实际标注（actual）来衡量预测标注（prediction）的好坏
func (e *FrapEvaluator) Evaluate(actual, prediction data.InstanceOutput) {
	if actual.Label == 0 {
		if prediction.Label == 0 {
			e.tn.Push(1.0)
			e.fn.Push(0.0)
			e.tp.Push(0.0)
			e.fp.Push(0.0)
		} else {
			e.tn.Push(0.0)
			e.fn.Push(0.0)
			e.tp.Push(0.0)
			e.fp.Push(1.0)
		}
	} else {
		if prediction.Label == 0 {
			e.tn.Push(0.0)
			e.fn.Push(1.0)
			e.tp.Push(0.0)
			e.fp.Push(0.0)
		} else {
			e.tn.Push(0.0)
			e.fn.Push(0.0)
			e.tp.Push(1.0)
			e.fp.Push(0.0)
		}
	}

	if actual.Label == prediction.Label {
		e.correct.Push(1.0)
	} else {
		e.correct.Push(0.0)
	}
}

// 返回最近size个（在Init中指定）样本的评价指标
func (e *FrapEvaluator) Report() eval.Evaluation {
	output := eval.Evaluation{}
	output.Metrics = make(map[string]float64)

	positive := (e.tp.Sum() + e.fp.Sum()) / float64(e.tp.NumValues())
	output.Metrics["positive"] = positive

	// 精度 = true-positive / (true-positive + false-positive)
	precision := e.tp.Sum() / (e.tp.Sum() + e.fp.Sum())
	output.Metrics["precision"] = precision

	// 召回率 = true-positive / (true-positive + false-negative)
	recall := e.tp.Sum() / (e.tp.Sum() + e.fn.Sum())
	output.Metrics["recall"] = recall

	// f1分值 = 2 * 精度 * 召回率 / (精度 + 召回率)
	fscore := 2 * precision * recall / (precision + recall)
	output.Metrics["fscore"] = fscore

	// 准确度 = 预测准确的样本数 / 总样本数
	accuracy := e.correct.Sum() / float64(e.correct.NumValues())
	output.Metrics["accuracy"] = accuracy

	return output
}
