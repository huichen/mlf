package data

import (
	"github.com/huichen/mlf/util"
)

// 部分数据集遍历器，见SkipDataset的注释
type SkipIterator struct {
	skips      []int
	skipsIndex int

	innerIterator DatasetIterator
}

func NewSkipIterator(set Dataset, skips []int) DatasetIterator {
	it := new(SkipIterator)
	it.innerIterator = set.CreateIterator()
	it.skips = skips
	it.skipsIndex = 0
	return it
}

func (it *SkipIterator) Start() {
	it.innerIterator.Start()
	it.innerIterator.Skip(it.skips[0])
	it.skipsIndex = 1
}

func (it *SkipIterator) End() bool {
	return it.innerIterator.End()
}

func (it *SkipIterator) Next() {
	if it.innerIterator.End() {
		return
	}
	it.innerIterator.Skip(it.skips[it.skipsIndex])
	it.skipsIndex = util.Mod(it.skipsIndex+1, len(it.skips))
	if it.skipsIndex == 0 {
		it.innerIterator.Skip(it.skips[it.skipsIndex])
		it.skipsIndex = 1
	}
}

func (it *SkipIterator) Skip(n int) {
	if it.innerIterator.End() {
		return
	}
	for i := 0; i < n; i++ {
		it.Next()
	}
}

func (it *SkipIterator) GetInstance() *Instance {
	return it.innerIterator.GetInstance()
}
