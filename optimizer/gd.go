package optimizer

import (
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/util"
	"log"
	"math"
)

// 梯度递降（Gradient Descent）优化器
type gdOptimizer struct {
	options OptimizerOptions
}

// 初始化优化结构体
func NewGdOptimizer(options OptimizerOptions) Optimizer {
	opt := new(gdOptimizer)
	opt.options = options
	return opt
}

// 清除结构体中保存的数据，以便重复使用结构体
func (opt *gdOptimizer) Clear() {
}

// 输入x_k和g_k，返回x需要更新的增量 d_k = - g_k
func (opt *gdOptimizer) GetDeltaX(x, g *util.Matrix) *util.Matrix {
	return g.Opposite()
}

func (opt *gdOptimizer) OptimizeWeights(
	weights *util.Matrix, derivative_func ComputeInstanceDerivativeFunc, set data.Dataset) {
	// 偏导数向量
	derivative := weights.Populate()

	// 学习率计算器
	learningRate := NewLearningRate(opt.options)

	// 优化循环
	iterator := set.CreateIterator()
	step := 0
	var learning_rate float64
	convergingSteps := 0
	oldWeights := weights.Populate()
	weightsDelta := weights.Populate()
	instanceDerivative := weights.Populate()
	log.Print("开始梯度递降优化")
	for {
		if opt.options.MaxIterations > 0 && step >= opt.options.MaxIterations {
			break
		}
		step++

		// 每次遍历样本前对偏导数向量清零
		derivative.Clear()

		// 遍历所有样本，计算偏导数向量并累加
		iterator.Start()
		instancesProcessed := 0
		for !iterator.End() {
			instance := iterator.GetInstance()
			derivative_func(weights, instance, instanceDerivative)
			derivative.Increment(instanceDerivative, 1.0/float64(set.NumInstances()))
			iterator.Next()
			instancesProcessed++

			if opt.options.GDBatchSize > 0 && instancesProcessed >= opt.options.GDBatchSize {
				// 添加正则化项
				derivative.Increment(ComputeRegularization(weights, opt.options),
					float64(instancesProcessed)/(float64(set.NumInstances())*float64(set.NumInstances())))

				// 计算特征权重的增量
				delta := opt.GetDeltaX(weights, derivative)

				// 根据学习率更新权重
				learning_rate = learningRate.ComputeLearningRate(delta)
				weights.Increment(delta, learning_rate)

				// 重置
				derivative.Clear()
				instancesProcessed = 0
			}
		}

		if instancesProcessed > 0 {
			// 处理剩余的样本
			derivative.Increment(ComputeRegularization(weights, opt.options),
				float64(instancesProcessed)/(float64(set.NumInstances())*float64(set.NumInstances())))
			delta := opt.GetDeltaX(weights, derivative)
			learning_rate = learningRate.ComputeLearningRate(delta)
			weights.Increment(delta, learning_rate)
		}

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
		if weightsDelta.Norm()/weights.Norm() < opt.options.ConvergingDeltaWeight {
			convergingSteps++
			if convergingSteps > opt.options.ConvergingSteps {
				log.Printf("收敛")
				break
			}
		}
	}
}
