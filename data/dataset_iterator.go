package data

// 数据集遍历器
type DatasetIterator interface {
	// 从头开始得到数据
	Start()

	// 是否抵达数据集的末尾
	End() bool

	// 跳转到下条数据
	Next()

	// 跳过n条数据，n>=0，当n为1时等价于Next()
	Skip(n int)

	// 得到当前数据样本
	// 请在调用前通过End()检查是否抵达数据集的末尾
	// 当访问失败或者End()为true时返回nil指针
	GetInstance() *Instance
}
