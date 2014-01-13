package online

import (
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/eval"
)

// 在线评价器接口
//
// 在线评价器内置循环缓冲区，假设缓冲区大小为N，当缓冲区满时，新数据
// 将从第一个（最老）的数据开始覆盖，这保证了我们总是评价最近N个数据样本。
type OnlineEvaluator interface {
	// 初始化并设置循环缓冲区的大小
	// 注意这会清除所有缓存的结果
	Init(size int)

	// 评价一条样本
	Evaluate(actual, prediction data.InstanceOutput)

	// 汇报评价结果
	Report() eval.Evaluation
}
