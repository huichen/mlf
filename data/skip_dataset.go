package data

import (
	"github.com/huichen/mlf/dictionary"
	"github.com/huichen/mlf/util"
)

// 跳过部分数据样本的数据集合
// SkipDataset从另一个数据集创建，创建方法见NewSkipDataset函数
type skipDataset struct {
	// 跳转信息
	// 使用第一个数据样本前先跳过skips[0]个样本
	// 然后在原始数据集中跳过skips[1]个样本，抵达第二个可访问的样本
	// 然后在原始数据集中跳过skips[2]个样本，抵达第三个可访问的样本
	// 以此类推
	skips []int

	// 原始数据集
	innerDataset Dataset

	// 数据集初始化选项
	initOptions DatasetOptions

	// 数据集中的样本总数
	numInstances int
}

type SkipBucket struct {
	SkipMode     bool
	NumInstances int
}

// 从另一个数据集创建数据集合
//
// buckets定义了怎样跳过数据集中的数据：
//   1. 当SkipMode为true时，跳过NumInstances个数据
//   2. 当SkipMode为false时，使用NumInstances个数据
// 重复buckets中的所有项，直到set遍历完为止。
func NewSkipDataset(set Dataset, buckets []SkipBucket) *skipDataset {
	skipSet := new(skipDataset)
	skipSet.innerDataset = set

	skip := 0
	for _, iBucket := range buckets {
		for iInst := 0; iInst < iBucket.NumInstances; iInst++ {
			if iBucket.SkipMode {
				skip++
			} else {
				skipSet.skips = append(skipSet.skips, skip)
				skip = 1
			}
		}
	}
	skipSet.skips = append(skipSet.skips, skip)

	skipSet.numInstances = 0
	numSkippedInstances := 0
	skipIndex := 0
	for {
		if numSkippedInstances >= set.NumInstances() {
			break
		}

		numSkippedInstances += skipSet.skips[skipIndex]
		skipSet.numInstances++

		skipIndex = util.Mod(skipIndex+1, len(skipSet.skips))
		if skipIndex == 0 {
			numSkippedInstances += skipSet.skips[skipIndex]
			skipIndex = 1
		}
	}
	skipSet.numInstances--

	return skipSet
}

func (set *skipDataset) NumInstances() int {
	return set.numInstances
}

func (set *skipDataset) CreateIterator() DatasetIterator {
	return NewSkipIterator(set.innerDataset, set.skips)
}

func (set *skipDataset) GetFeatureDictionary() *dictionary.Dictionary {
	return set.innerDataset.GetFeatureDictionary()
}

func (set *skipDataset) GetLabelDictionary() *dictionary.Dictionary {
	return set.innerDataset.GetLabelDictionary()
}

func (set *skipDataset) GetOptions() DatasetOptions {
	return set.innerDataset.GetOptions()
}
