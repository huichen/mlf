package online

import (
	"github.com/huichen/mlf/optimizer"
)

type OnlineSGDClassifierOptions struct {
	// 分类数目
	NumLabels int

	// 每接收多少个样本进行一次参数更新，当值为0时（默认）设为1
	BatchSize int

	// 优化器选项
	Optimizer optimizer.OptimizerOptions

	// 对最近的多少个样本进行模型评估
	NumInstancesForEvaluation int
}
