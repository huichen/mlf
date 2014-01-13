package data

import (
	"github.com/huichen/mlf/util"
)

// 数据样本的输出
type InstanceOutput struct {
	// 标注，用于分类问题
	Label int

	// 目标函数的值，用于回归问题
	Value float64

	// 标注的字符串
	LabelString string

	// 标注分布，用于分类问题的输出（各分类的概率分布）
	LabelDistribution *util.Vector
}
