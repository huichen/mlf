package eval

import (
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/supervised"
)

// Accuracy evaluator
type AccuracyEvaluator struct {
}

func (e *AccuracyEvaluator) Evaluate(m supervised.Model, set data.Dataset) (result Evaluation) {
	correctPrediction := 0
	totalPrediction := 0

	iter := set.CreateIterator()
	iter.Start()
	for !iter.End() {
		instance := iter.GetInstance()
		out := m.Predict(instance)
		if instance.Output.LabelString == out.LabelString {
			correctPrediction++
		}
		totalPrediction++
		iter.Next()
	}

	result.Metrics = make(map[string]float64)
	result.Metrics["accuracy"] = float64(correctPrediction) / float64(totalPrediction)

	return
}
