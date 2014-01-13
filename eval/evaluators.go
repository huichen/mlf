package eval

import (
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/supervised"
)

// 包含一个评价器的集合
type Evaluators struct {
	evaluators []Evaluator
}

func NewEvaluators(evaluators []Evaluator) *Evaluators {
	evals := new(Evaluators)
	evals.evaluators = evaluators
	return evals
}

func (evals *Evaluators) Evaluate(m supervised.Model, set data.Dataset) Evaluation {
	output := Evaluation{}
	output.Metrics = make(map[string]float64)

	for _, e := range evals.evaluators {
		result := e.Evaluate(m, set)
		for name, value := range result.Metrics {
			output.Metrics[name] = value
		}
	}
	return output
}
