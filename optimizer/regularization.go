package optimizer

import (
	"github.com/huichen/mlf/util"
)

// 根据正则化方法计算偏导数向量需要添加正则化项
func ComputeRegularization(weights *util.Matrix, options OptimizerOptions) *util.Matrix {
	reg := weights.Populate()

	if options.RegularizationScheme == 1 {
		// L-1正则化
		for iLabel := 0; iLabel < weights.NumLabels(); iLabel++ {
			for _, k := range weights.GetValues(iLabel).Keys() {
				if weights.Get(iLabel, k) > 0 {
					reg.Set(iLabel, k, options.RegularizationFactor)
				} else {
					reg.Set(iLabel, k, -options.RegularizationFactor)
				}
			}
		}
	} else if options.RegularizationScheme == 2 {
		// L-2正则化
		for iLabel := 0; iLabel < weights.NumLabels(); iLabel++ {
			for _, k := range weights.GetValues(iLabel).Keys() {
				reg.Set(iLabel, k, options.RegularizationFactor*weights.Get(iLabel, k))
			}
		}
	}

	return reg
}
