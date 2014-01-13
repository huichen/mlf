package optimizer

import (
	"fmt"
	"github.com/huichen/mlf/util"
	"testing"
)

func TestLbfgsOptimizer(t *testing.T) {
	opt := NewLbfgsOptimizer(OptimizerOptions{})

	x := util.NewMatrix(1, 2)
	g := util.NewMatrix(1, 2)
	x.GetValues(0).SetValues([]float64{1, 0.3})

	k := 0
	for {
		g.GetValues(0).SetValues([]float64{4 * x.Get(0, 0) * x.Get(0, 0) * x.Get(0, 0), 4 * x.Get(0, 1) * x.Get(0, 1) * x.Get(0, 1)})
		delta := opt.GetDeltaX(x, g)
		x.Increment(delta, 1)
		k++
		if delta.Norm() < 0.0001 {
			break
		}
	}

	fmt.Println("==== LBFGS优化完成 ====")
	fmt.Println("循环数", k)
	fmt.Println("x = ", x)
}
