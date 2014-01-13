package data

import (
	"log"
)

// 内存数据集遍历器
type inmemDatasetIterator struct {
	set       *inmemDataset
	currIndex int
}

func (it *inmemDatasetIterator) Start() {
	it.set.CheckFinalized(true)
	it.currIndex = 0
}

func (it *inmemDatasetIterator) End() bool {
	it.set.CheckFinalized(true)
	if it.currIndex >= len(it.set.instances) {
		return true
	}
	return false
}

func (it *inmemDatasetIterator) Next() {
	it.set.CheckFinalized(true)
	if !it.End() {
		it.currIndex++
	}
}

func (it *inmemDatasetIterator) Skip(n int) {
	it.set.CheckFinalized(true)
	if n < 0 {
		log.Fatal("Skip参数必须大于等于0")
	}
	it.currIndex += n
}

func (it *inmemDatasetIterator) GetInstance() *Instance {
	it.set.CheckFinalized(true)
	if it.End() {
		return nil
	}
	return it.set.instances[it.currIndex]
}
