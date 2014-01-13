最大熵分类器
====

最大熵分类器（maximum entropy classifier，也称作log-linear classifier）是业界最流行的机器学习分类模型。这种模型的优点有

* 训练的时间和空间复杂度为O(n)，其中n为样本数，因此适合处理大数据
* 适合做特征工程（feature engineering），容易解释模型中特征的权重
* 能够处理多分类问题，当分类数为2时退化成logisitic回归
* 在训练样本充足（至少比特征数多一个数量级）的情况下，性能（精度、召回率等）和其它最好的模型相近
* 结合stochastic gradient descent后可[在线学习](/doc/online.md)，是当前点击预测的主流模型

所以如果你只要学习一种分类模型的话，最大熵分类模型值得你花时间好好研究一下，比如阅读[这篇](http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92)和[这篇](http://en.wikipedia.org/wiki/Multinomial_logistic_regression)。

弥勒佛框架中完全实现了最大熵分类模型，并提供了一个瑞士军刀程序[tool/classifier.go](/tool/classifier.go)让你从命令行对数据文件进行建模和评价。在这篇文章中，我们结合这个工具来阐述怎么进行最大熵分类。

在开始分析代码前，请先尝试运行这个程序：先进入/tool/文件夹，然后运行

```
go run classifier.go --input ../testdata/a1a --folds 5
```

这完成了对a1a文件中数据的5-fold交叉评价，评价结果会在输出的最后打印出来，比如

```
5-folds 交叉评价：
精度 = 70.39 %
召回率 = 55.81 %
F1 = 62.14 %
准确度 = 83.30 %
```

a1a中数据的输出是一个成年人的年收入是否超过5万美元，输入的特征是该成年人的职业、种族、教育等属性。我们可以看到，模型预测的精度达到了70%。

请自行尝试[testdata目录](/testdata/)下其它的数据集（来自[这个网页](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)），并试着改变命令行参数的值来改变收敛速度，比如使用GD优化器，或者其它的正则化方法和学习率等。有时你会发现训练不收敛甚至发散，可以通过降低学习率（--learning_rate命令行参数）来解决。

在尝试几次后，你对这个程序的功能就有了大致了解，下面我们一步步分析这个程序是怎么实现的。

## 参数定义

首先我们需要从命令行输入一些训练使用的参数：

```go
// 数据输入
libsvm_file = flag.String("input", "", "libsvm格式的数据文件，训练数据")
test_file = flag.String("test", "", "libsvm格式的数据文件，测试数据")

// 模型输出
model_file = flag.String("output", "model.mlf", "模型输出")

// 机器学习参数
opt = flag.String("optimizer", "lbfgs", "优化器")
reg = flag.Int("regularization", 2, "正则化方法")
reg_factor = flag.Float64("reg_factor", float64(1), "正则化因子")
learning_rate = flag.Float64("learning_rate", float64(1), "学习率")
characteristic_time = flag.Float64("characteristic_time", float64(0), "学习率特征时间")
batch_size = flag.Int("batch_size", 0, "梯度递降法的batch尺寸: 0为full batch, 1为stochastic, 其它值为mini batch")
delta = flag.Float64("delta", 1e-3, "权重变化量和权重的比值(|dw|/|w|)小于此值时判定为收敛")
max_iter = flag.Int("max_iter", 0, "优化器最多迭代多少次")
folds = flag.Int("folds", 0, "N-交叉评价，值为零时不交叉评价")
```

其中定义了模型的输入输出，[优化器](/doc/optimizer.md)，[学习率](/doc/optimizer.md#学习率)和[正则化方法](/doc/optimizer.md#正则化)等参数。弥勒佛框架提供了丰富的参数可供用户调节，方便用户在不同类型的数据上都能达到最理想的训练效果。

## 载入训练数据

训练之前我们需要载入训练数据，这里提供了一个库可以让你从libsvm格式的文件中载入数据集：

```go
// 载入训练集
set := contrib.LoadLibSVMDataset(*libsvm_file, false)
```

LoadLibSVMDataset的用法请见[contrib/libsvm_dataset_loader.go](/contrib/libsvm_dataset_loader.go)。和LoadLibSVMDataset对应的，弥勒佛也提供了一个SaveLibSVMDataset函数将数据集保存到libsvm格式的文件中，见[contrib/libsvm_dataset_saver.go](/contrib/libsvm_dataset_saver.go)文件。

你也可以逐条地载入训练样本，详情见[数据集文档](/doc/dataset.md)。

## 创建训练器

创建训练器的代码如下

```go
// 设置训练器参数
trainerOptions := supervised.TrainerOptions{
	Optimizer: optimizer.OptimizerOptions{
		OptimizerName:         *opt,
		RegularizationScheme:  *reg,
		RegularizationFactor:  *reg_factor,
		LearningRate:          *learning_rate,
		CharacteristicTime:    *characteristic_time,
		ConvergingDeltaWeight: *delta,
		ConvergingSteps:       3,
		MaxIterations:         *max_iter,
		GDBatchSize:           *batch_size,
	}}
// 创建训练器
trainer := supervised.NewMaxEntClassifierTrainer(trainerOptions)
```

其中trainerOptions中主要定义了优化器需要的参数，具体请见[优化器文档](/doc/optimizer.md)。

## 交叉评价

[交叉评价](/doc/cross_validate.md)(cross-validate)是一种调参数的方法，该方法将训练集和检验集区分开来以避免过拟合。可以使用如下代码实现交叉评价：

```go
// 进行交叉评价
evaluators := eval.NewEvaluators([]eval.Evaluator{
	&eval.PREvaluator{}, &eval.AccuracyEvaluator{}})
if *folds != 0 {
	result := eval.CrossValidate(trainer, set, evaluators, *folds)
	log.Print(*folds, "-folds 交叉评价：")
	log.Printf("精度   =  %.2f %%", result.Metrics["precision"]*100)
	log.Printf("召回率 =  %.2f %%", result.Metrics["recall"]*100)
	log.Printf("F1     =  %.2f %%", result.Metrics["fscore"]*100)
	log.Printf("准确度 =  %.2f %%", result.Metrics["accuracy"]*100)
}
```

其中evaluators是一个包含多个评价器的切片，关于评价器，请见[文档](/doc/eval.md)。

## 训练、保存和载入模型

交叉评价完成参数学习之后，通常我们会在全部训练集上重新训练一次得到一个最好的模型。训练的代码非常简单：

```go
// 在全部数据上训练模型
model := trainer.Train(set)
model.Write(*model_file)
```

模型文件会被写入到model_file指定的路径中去，这个文件实际上是json格式的，容易阅读。

训练出的模型model满足下面定义的通用分类器接口

```go
// 训练得到的机器学习模型
type Model interface {
	// 返回模型类型，比如"maxent_classifier"
	GetModelType() string

	// 将模型写入文件
	Write(path string)

	// 预测样本的输出
	Predict(instance *data.Instance) data.InstanceOutput
}
```

因此你可以使用Predict函数对一个未知的训练样本进行预测。载入一个模型请使用[supervised/model_loader.go](/supervised/model_loader.go)中定义的LoadModel函数：

```go
func LoadModel(path string) Model
```

这个函数会自动识别模型的类型。

## 测试集检验

通常我们会保留一部分数据（holdout set）用以最终检验不同模型的好坏。可以使用下面的代码进行测试集检验。

```go
// 测试模型
if *test_file != "" {
	// 载入测试集
	testSet := contrib.LoadLibSVMDataset(*test_file, false)

	// 在测试集上评价模型并输出结果
	result := evaluators.Evaluate(model, testSet)
	log.Print("测试数据集评价：")
	log.Printf("精度   =  %.2f %%", result.Metrics["precision"]*100)
	log.Printf("召回率 =  %.2f %%", result.Metrics["recall"]*100)
	log.Printf("F1     =  %.2f %%", result.Metrics["fscore"]*100)
	log.Printf("准确度 =  %.2f %%", result.Metrics["accuracy"]*100)
}
```

## 总结

这篇文章讨论了怎样使用最大熵模型对训练数据进行建模、交叉评价、测试集检验和模型的文件读写。所有的这些功能都已经在弥勒佛框架中实现，可以作为库嵌入到程序中使用，也提供了独立的程序可在命令行直接运行。

上面的讨论适合训练小于千万规模的数据，如果你的训练集很大无法载入到一台机器的内存中，有几种处理方法：

* 抽样，将数据规模降低到百万量级。
* 将数据裂分成N份然后在N台机器上训练，最终把生成的N个最大熵模型合并为一个集成模型（ensemble model）。
* 使用非本地数据集，从网络、数据库和文件系统中逐批载入训练数据进行训练，见[数据集文档](/doc/dataset.md#扩展)。
* stream数据进行[在线学习](/doc/online.md)。
