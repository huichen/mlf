优化器
====

优化器的目标是为了寻找使得loss function最小的模型参数。

优化器的接口如下：

```go
type Optimizer interface {
        Clear()
        GetDeltaX(x, g *util.Matrix) *util.Matrix
        OptimizeWeights(weights *util.Matrix,
                derivative_func ComputeInstanceDerivativeFunc, set data.Dataset)
}
```

其中

* 因为优化器可能存储了一些局部变量，因此在每次优化任务开始前必须调用Clear()函数清空这些局部变量
* GetDeltaX函数根据当前的参数值x和loss function的偏导数g决定参数需要调整的增量
* OptimizeWeights函数通过调用GetDeltaX对weights进行多次调整得到最优weights，这需要计算偏导数函数derivative_func

我们定义了两中优化器，l-BFGS和梯度递降（Gradient Descent）。可以通过下面的函数来创建这两种优化器

```go
func NewOptimizer(options OptimizerOptions) Optimizer
```

其中options定义了优化器选项

```go
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
```

其中
* 当OptimizerName为"lbfgs"时创建lbfgs优化器，为"gd"时创建梯度递降优化器
* lbfgs通常比梯度递降需要的迭代数小一个数量级，而且lbfgs使用了协程并发极大加快了计算速度，因此推荐使用
* 当使用GD优化器时，GDBatchSize为0则为full-batch GD，GDBatchSize为1时为stocastic GD，GDBatchSize为n(n>1)时为mini-batch GD

## 学习率

学习率决定了每次优化参数时参数变化的增量大小，学习率过小会导致更长的收敛时间，学习率过大可能会导致震荡不收敛甚至是发散到无穷大。

OptimizerOptions中的LearningRate和CharacteristicTime定义了带annealing的学习率：

```alpha = LearningRate / (1 + t / CharacteristicTime)```

其中
* t为循环数（从0开始）
* LearningRate为初始的学习率，如果值为0则使用学习率=1
* 当CharacteristicTime为0时我们使用常数学习率LearningRate

## 正则化

适当的正则化可以避免过拟合(overfitting)或者可以帮助降低特征维度。弥勒佛框架实现了L1正则化和L2正则化。正则化方法选择由OptimizerOptions中的两个参数控制：

```go
// 正则化方法：
// 当值为 0 时，不使用正则化
// 当值为 1 时，使用L1正则化
// 当值为 2 时，使用L2正则化
RegularizationScheme int

// 正则化因子
RegularizationFactor float64
```
