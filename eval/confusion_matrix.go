package eval

import (
	"fmt"
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/supervised"
)

// 输出模型的混淆矩阵
type ConfusionMatrixEvaluator struct {
}

// 输出的度量名字为 "confusion:M/N" 其中M为真实标注，N为预测标注
func (e *ConfusionMatrixEvaluator) Evaluate(m supervised.Model, set data.Dataset) (result Evaluation) {
	result.Metrics = make(map[string]float64)
	iter := set.CreateIterator()
	iter.Start()
	for !iter.End() {
		instance := iter.GetInstance()
		out := m.Predict(instance)
		name := fmt.Sprintf("confusion:%d/%d", instance.Output.Label, out.Label)
		result.Metrics[name]++
		iter.Next()
	}
	return
}
