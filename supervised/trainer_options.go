package supervised

import (
	"github.com/huichen/mlf/optimizer"
)

// 训练器选项
type TrainerOptions struct {
	// 优化器选项
	Optimizer optimizer.OptimizerOptions

	// 其它自定义选项
	Options interface{}
}
