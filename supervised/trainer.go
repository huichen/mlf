package supervised

import (
	"github.com/huichen/mlf/data"
)

type Trainer interface {
	// 在数据集上进行训练，得到模型
	Train(set data.Dataset) Model
}
