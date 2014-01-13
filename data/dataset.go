package data

import (
	"github.com/huichen/mlf/dictionary"
)

// Dataset定义了数据集的访问借口
//
// 数据集的访问有下面两个特点：
// 1. 重复访问保证顺序一致。
// 2. 可以确保每个不同的数据样本（Instance）会被访问一次且仅被访问一次。
type Dataset interface {
	// 数据集中的样本数目
	NumInstances() int

	// 新建一个遍历器
	CreateIterator() DatasetIterator

	// 得到数据集参数
	GetOptions() DatasetOptions

	// 得到特征词典
	// 当不使用特征词典时（直接输入整数特征ID）返回nil
	GetFeatureDictionary() *dictionary.Dictionary

	// 得到标注词典
	// 当不使用标注词典时（直接输入整数Label时）返回nil
	GetLabelDictionary() *dictionary.Dictionary
}
