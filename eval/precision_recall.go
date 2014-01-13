package eval

import (
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/supervised"
	"log"
)

// Precision-recall-accuracy evaluator
// 仅当模型是二分类问题时输出有意义
type PREvaluator struct {
}

func (e *PREvaluator) Evaluate(m supervised.Model, set data.Dataset) (result Evaluation) {
	tp := 0 // true-positive
	tn := 0 // true-negative
	fp := 0 // false-positive
	fn := 0 // false-negative

	iter := set.CreateIterator()
	iter.Start()
	for !iter.End() {
		instance := iter.GetInstance()
		if instance.Output.Label > 2 {
			log.Fatal("调用PREvaluator但不是二分类问题")
		}

		out := m.Predict(instance)
		if out.Label == 0 {
			if instance.Output.Label == 0 {
				tn++
			} else {
				fn++
			}
		} else {
			if instance.Output.Label == 0 {
				fp++
			} else {
				tp++
			}
		}
		iter.Next()
	}

	result.Metrics = make(map[string]float64)
	result.Metrics["precision"] = float64(tp) / float64(tp+fp)
	result.Metrics["recall"] = float64(tp) / float64(tp+fn)
	result.Metrics["tp"] = float64(tp)
	result.Metrics["fp"] = float64(fp)
	result.Metrics["tn"] = float64(tn)
	result.Metrics["fn"] = float64(fn)
	result.Metrics["fscore"] =
		2 * result.Metrics["precision"] * result.Metrics["recall"] / (result.Metrics["precision"] + result.Metrics["recall"])

	return
}
