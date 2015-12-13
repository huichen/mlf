package main

import (
	"flag"
	"github.com/huichen/mlf/contrib"
	"github.com/huichen/mlf/rbm"
	"runtime"
)

var (
	// 数据输入
	libsvm_file = flag.String("input", "", "libsvm格式的数据文件，训练数据")
	model       = flag.String("model", "model.mlf", "写入的模型文件")

	// 机器学习参数
	learning_rate = flag.Float64("learning_rate", 0.01, "学习率")
	batch_size    = flag.Int("batch_size", 100,
		"梯度递降法的batch尺寸: 1为stochastic, 其它值为mini batch")
	delta = flag.Float64("delta", 1e-4,
		"权重变化量和权重的比值(|dw|/|w|)小于此值时判定为收敛")
	maxIter   = flag.Int("max_iter", 0, "优化器最多迭代多少次")
	hidden    = flag.Int("hidden", 10, "多少个隐藏单元")
	numCD     = flag.Int("cd", 1, "CD次数")
	useBinary = flag.Bool("binary_hidden", true, "是否使用抽样隐藏单元")
)

func main() {
	flag.Parse()
	runtime.GOMAXPROCS(runtime.NumCPU())

	// 载入训练集
	set := contrib.LoadLibSVMDataset(*libsvm_file, false)

	options := rbm.RBMOptions{
		NumHiddenUnits:       *hidden,
		NumCD:                *numCD,
		Worker:               runtime.NumCPU(),
		LearningRate:         *learning_rate,
		MaxIter:              *maxIter,
		BatchSize:            *batch_size,
		Delta:                *delta,
		UseBinaryHiddenUnits: *useBinary,
	}

	// 创建训练器
	machine := rbm.NewRBM(options)

	machine.Train(set)

	machine.Write(*model)
}
