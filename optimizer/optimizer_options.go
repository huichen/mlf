package optimizer

type OptimizerOptions struct {
	// 优化器名称
	OptimizerName string

	// 正则化方法：
	// 当值为 0 时，不使用正则化
	// 当值为 1 时，使用L1正则化
	// 当值为 2 时，使用L2正则化
	RegularizationScheme int

	// 正则化因子
	// 当值为0.0值，使用 --default_regularization_factor
	RegularizationFactor float64

	// 下面两个参数定义带Annealing的学习率：
	//
	// alpha = LearningRate / (1 + t / CharacteristicTime)
	//
	// 其中
	// 1. t为循环数（从0开始）
	// 2. LearningRate为初始的学习率，如果值为0则使用学习率=1
	// 3. 当CharacteristicTime为0时我们使用常数学习率LearningRate
	LearningRate       float64
	CharacteristicTime float64

	// 最多执行多少次优化循环，当值为0时不设限
	MaxIterations int

	// 收敛条件：
	// 1. |dw| / |w| < ConvergingDeltaWeight
	// 2. 必须连续满足上一个条件NumConvergingSteps次
	ConvergingDeltaWeight float64
	ConvergingSteps       int

	// 每批次更新权重前累加多少个样本的的偏导数（仅对GD优化器起作用）
	// GDBatchSize = 0         // full-bath gradient descent
	// GDBatchSize = 1         // stochastic gradient descent
	// GDBatchSize = n (n>1)   // mini-batch gradient descent
	GDBatchSize int
}
