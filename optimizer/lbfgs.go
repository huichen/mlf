package optimizer

import (
	"flag"
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/util"
	"log"
	"math"
	"runtime"
)

var (
	lbfgs_history_size = flag.Int("lbfgs_history_size", 5, "L-BFGS中存储的历史步长")
	lbfgs_threads      = flag.Int("lbfgs_threads", 0, "使用多少个协程进行LBFGS收敛，值为0时使用所有CPU")
)

// limited-memory BFGS优化器
//
// l-bfgs的迭代算法见下面的论文
//   Nocedal, J. (1980). "Updating Quasi-Newton Matrices with Limited Storage".
//   Mathematics of Computation 35 (151): 773–782. doi:10.1090/S0025-5718-1980-0572855-7
//
// 这种方法最多保存最近m步的中间结果用以计算海森矩阵的近似值。
//
// 请用NewOptimizer函数建立新的优化器。
//
// 注意optimizer是
//   1. 协程不安全的，请为每个协程建一个optimizer；
//   2. 迭代不安全的，optimizer会记录每个迭代步骤的中间结果，如果需要重新开始新的优化，
//      请调用Clear函数。
type lbfgsOptimizer struct {
	// 初始化参数
	options OptimizerOptions

	// 当前的步数，从0开始
	// 如果需要重新优化，请调用Clear函数
	k int

	// 自变量
	x []*util.Matrix

	// 目标函数的偏导数向量
	g []*util.Matrix

	// s_k = x_(k+1) - x_k
	s []*util.Matrix

	// y_k = g_(k+1) - g_k
	y []*util.Matrix

	// ro_k = 1 / Y_k .* s_k
	ro *util.Vector

	// 特征向量维度
	labels int

	// 临时变量
	q, z        *util.Matrix
	alpha, beta *util.Vector
}

// 开辟新的lbfgsOptimizer指针
func NewLbfgsOptimizer(options OptimizerOptions) Optimizer {
	opt := new(lbfgsOptimizer)
	opt.options = options
	opt.k = 0
	return opt
}

// 初始化优化结构体
// 为结构体中的向量分配新的内存，向量的长度可能发生变化。
func (opt *lbfgsOptimizer) initStruct(labels, features int, isSparse bool) {
	opt.labels = labels

	opt.x = make([]*util.Matrix, *lbfgs_history_size)
	opt.g = make([]*util.Matrix, *lbfgs_history_size)
	opt.s = make([]*util.Matrix, *lbfgs_history_size)
	opt.y = make([]*util.Matrix, *lbfgs_history_size)

	opt.ro = util.NewVector(*lbfgs_history_size)
	opt.alpha = util.NewVector(*lbfgs_history_size)
	opt.beta = util.NewVector(*lbfgs_history_size)
	if !isSparse {
		opt.q = util.NewMatrix(labels, features)
		opt.z = util.NewMatrix(labels, features)
		for i := 0; i < *lbfgs_history_size; i++ {
			opt.x[i] = util.NewMatrix(labels, features)
			opt.g[i] = util.NewMatrix(labels, features)
			opt.s[i] = util.NewMatrix(labels, features)
			opt.y[i] = util.NewMatrix(labels, features)
		}
	} else {
		opt.q = util.NewSparseMatrix(labels)
		opt.z = util.NewSparseMatrix(labels)
		for i := 0; i < *lbfgs_history_size; i++ {
			opt.x[i] = util.NewSparseMatrix(labels)
			opt.g[i] = util.NewSparseMatrix(labels)
			opt.s[i] = util.NewSparseMatrix(labels)
			opt.y[i] = util.NewSparseMatrix(labels)
		}
	}
}

// 清除结构体中保存的数据，以便重复使用结构体
func (opt *lbfgsOptimizer) Clear() {
	opt.k = 0
}

// 输入x_k和g_k，返回x需要更新的增量 d_k = - H_k * g_k
func (opt *lbfgsOptimizer) GetDeltaX(x, g *util.Matrix) *util.Matrix {
	if x.NumLabels() != g.NumLabels() {
		log.Fatal("x和g的维度不一致")
	}

	// 第一次调用时开辟内存
	if opt.k == 0 {
		if x.IsSparse() {
			opt.initStruct(x.NumLabels(), 0, x.IsSparse())
		} else {
			opt.initStruct(x.NumLabels(), x.NumValues(), x.IsSparse())
		}
	}

	currIndex := util.Mod(opt.k, *lbfgs_history_size)

	// 更新x_k
	opt.x[currIndex].DeepCopy(x)

	// 更新g_k
	opt.g[currIndex].DeepCopy(g)

	// 当为第0步时，使用简单的gradient descent
	if opt.k == 0 {
		opt.k++
		return g.Opposite()
	}

	prevIndex := util.Mod(opt.k-1, *lbfgs_history_size)

	// 更新s_(k-1)
	opt.s[prevIndex].WeightedSum(opt.x[currIndex], opt.x[prevIndex], 1, -1)

	// 更新y_(k-1)
	opt.y[prevIndex].WeightedSum(opt.g[currIndex], opt.g[prevIndex], 1, -1)

	// 更新ro_(k-1)
	opt.ro.Set(prevIndex, 1.0/util.MatrixDotProduct(opt.y[prevIndex], opt.s[prevIndex]))

	// 计算两个循环的下限
	lowerBound := opt.k - *lbfgs_history_size
	if lowerBound < 0 {
		lowerBound = 0
	}

	// 第一个循环
	opt.q.DeepCopy(g)
	for i := opt.k - 1; i >= lowerBound; i-- {
		currIndex := util.Mod(i, *lbfgs_history_size)
		opt.alpha.Set(currIndex,
			opt.ro.Get(currIndex)*util.MatrixDotProduct(opt.s[currIndex], opt.q))
		opt.q.Increment(opt.y[currIndex], -opt.alpha.Get(currIndex))
	}

	// 第二个循环
	opt.z.DeepCopy(opt.q)
	for i := lowerBound; i <= opt.k-1; i++ {
		currIndex := util.Mod(i, *lbfgs_history_size)
		opt.beta.Set(currIndex,
			opt.ro.Get(currIndex)*util.MatrixDotProduct(opt.y[currIndex], opt.z))
		opt.z.Increment(opt.s[currIndex],
			opt.alpha.Get(currIndex)-opt.beta.Get(currIndex))
	}

	// 更新k
	opt.k++

	return opt.z.Opposite()
}

func (opt *lbfgsOptimizer) OptimizeWeights(
	weights *util.Matrix, derivative_func ComputeInstanceDerivativeFunc, set data.Dataset) {

	// 学习率计算器
	learningRate := NewLearningRate(opt.options)

	// 偏导数向量
	derivative := weights.Populate()

	// 优化循环
	step := 0
	convergingSteps := 0
	oldWeights := weights.Populate()
	weightsDelta := weights.Populate()

	// 为各个工作协程开辟临时资源
	numLbfgsThreads := *lbfgs_threads
	if numLbfgsThreads == 0 {
		numLbfgsThreads = runtime.NumCPU()
	}
	workerSet := make([]data.Dataset, numLbfgsThreads)
	workerDerivative := make([]*util.Matrix, numLbfgsThreads)
	workerInstanceDerivative := make([]*util.Matrix, numLbfgsThreads)
	for iWorker := 0; iWorker < numLbfgsThreads; iWorker++ {
		workerBuckets := []data.SkipBucket{
			{true, iWorker},
			{false, 1},
			{true, numLbfgsThreads - 1 - iWorker},
		}
		workerSet[iWorker] = data.NewSkipDataset(set, workerBuckets)
		workerDerivative[iWorker] = weights.Populate()
		workerInstanceDerivative[iWorker] = weights.Populate()
	}

	log.Print("开始L-BFGS优化")
	for {
		if opt.options.MaxIterations > 0 && step >= opt.options.MaxIterations {
			break
		}
		step++

		// 开始工作协程
		workerChannel := make(chan int, numLbfgsThreads)
		for iWorker := 0; iWorker < numLbfgsThreads; iWorker++ {
			go func(iw int) {
				workerDerivative[iw].Clear()
				iterator := workerSet[iw].CreateIterator()
				iterator.Start()
				for !iterator.End() {
					instance := iterator.GetInstance()
					derivative_func(
						weights, instance, workerInstanceDerivative[iw])
					//					log.Print(workerInstanceDerivative[iw].GetValues(0))
					workerDerivative[iw].Increment(
						workerInstanceDerivative[iw], float64(1)/float64(set.NumInstances()))
					iterator.Next()
				}
				workerChannel <- iw
			}(iWorker)
		}

		derivative.Clear()

		// 等待工作协程结束
		for iWorker := 0; iWorker < numLbfgsThreads; iWorker++ {
			<-workerChannel
		}
		for iWorker := 0; iWorker < numLbfgsThreads; iWorker++ {
			derivative.Increment(workerDerivative[iWorker], 1)
		}

		// 添加正则化项
		derivative.Increment(ComputeRegularization(weights, opt.options), 1.0/float64(set.NumInstances()))

		// 计算特征权重的增量
		delta := opt.GetDeltaX(weights, derivative)

		// 根据学习率更新权重
		learning_rate := learningRate.ComputeLearningRate(delta)
		weights.Increment(delta, learning_rate)

		weightsDelta.WeightedSum(weights, oldWeights, 1, -1)
		oldWeights.DeepCopy(weights)
		weightsNorm := weights.Norm()
		weightsDeltaNorm := weightsDelta.Norm()
		log.Printf("#%d |dw|/|w|=%f |w|=%f lr=%1.3g", step, weightsDeltaNorm/weightsNorm, weightsNorm, learning_rate)

		// 判断是否溢出
		if math.IsNaN(weightsNorm) {
			log.Fatal("优化失败：不收敛")
		}

		// 判断是否收敛
		if weightsDeltaNorm/weightsNorm < opt.options.ConvergingDeltaWeight {
			convergingSteps++
			if convergingSteps > opt.options.ConvergingSteps {
				log.Printf("收敛")
				break
			}
		} else {
			convergingSteps = 0
		}
	}
}
