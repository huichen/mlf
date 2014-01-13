package eval

import (
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/supervised"
)

type Evaluator interface {
	Evaluate(m supervised.Model, set data.Dataset) Evaluation
}
