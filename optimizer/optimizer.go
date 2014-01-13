package optimizer

import (
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/util"
	"log"
)

type ComputeInstanceDerivativeFunc func(
	weights *util.Matrix, instance *data.Instance, instanceDerivative *util.Matrix)

// 通用优化器接口
type Optimizer interface {
	Clear()
	GetDeltaX(x, g *util.Matrix) *util.Matrix
	OptimizeWeights(weights *util.Matrix,
		derivative_func ComputeInstanceDerivativeFunc, set data.Dataset)
}

func NewOptimizer(options OptimizerOptions) Optimizer {
	if options.OptimizerName == "lbfgs" {
		return NewLbfgsOptimizer(options)
	} else if options.OptimizerName == "gd" {
		return NewGdOptimizer(options)
	}

	log.Fatal("必须指定合法的OptimizerName")
	return nil
}
