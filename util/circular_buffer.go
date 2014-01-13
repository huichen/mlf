package util

// 环形缓存
// 数据从0位置开始添加，当溢出时指针重置为0，新数据从0位置开始覆盖
type CircularBuffer struct {
	values   []float64
	index    int
	overflow bool
}

// 必须使用该函数创建环形缓存，size为缓存区长度
func NewCircularBuffer(size int) *CircularBuffer {
	cb := new(CircularBuffer)
	cb.values = make([]float64, size)
	cb.index = 0
	cb.overflow = false
	return cb
}

// 返回环形缓存中的实际存储的数值个数
func (cb *CircularBuffer) NumValues() int {
	if !cb.overflow {
		return cb.index
	}
	return len(cb.values)
}

// 向缓存中添加一个数
func (cb *CircularBuffer) Push(v float64) {
	cb.values[cb.index] = v
	cb.index++
	if cb.index >= len(cb.values) {
		cb.index = 0
		cb.overflow = true
	}
}

// 计算缓存中所有数之和
func (cb *CircularBuffer) Sum() (s float64) {
	if cb.overflow {
		for _, v := range cb.values {
			s += v
		}
	} else {
		for i := 0; i < cb.index; i++ {
			s += cb.values[i]
		}
	}
	return
}
