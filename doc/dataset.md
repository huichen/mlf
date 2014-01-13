数据集
====

定义数据集（[Dataset](/data/dataset.go)）的意义在于统一机器学习数据的调用接口，无论是监督式学习，非监督式学习还是在线学习问题都可以使用该接口访问数据。也就是说数据只要收集整理一次，就可以用于所有弥勒佛框架中的模型，这为我们快速尝试不同模型提供了极大便利。

数据集是一个抽象的接口，定义如下

```go
type Dataset interface {
	// 数据集中的样本数目
	NumInstances() int
	
	// 新建一个遍历器
	CreateIterator() DatasetIterator
	
	// 得到数据集参数
	GetOptions() DatasetOptions

	// 得到特征词典
	// 当不使用特征词典时（直接输入整数特征ID）返回nil
	GetFeatureDictionary() *dictionary.Dictionary

	// 得到标注词典
	// 当不使用标注词典时（直接输入整数Label时）返回nil
	GetLabelDictionary() *dictionary.Dictionary
}
```

数据集需要通过数据集遍历器（[DatasetIterator](/data/dataset_iterator.go)）来访问，数据集遍历器也是一个抽象的接口：

```go
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
```

可以根据数据集存储的媒介和访问方式的不同实现不同的数据集及其遍历器，弥勒佛框架中有两个具体的实现（内存存储数据集和跳跃数据集）。在解释这两个实现前，我们需要了解数据集参数。

## 数据集参数

数据集参数描述了数据集中数据的类型和格式，见下面的代码和注释。

```go
type DatasetOptions struct {
	// 特征是否使用稀疏向量存储
	FeatureIsSparse bool

	// 特征维度，仅当FeatureIsSparse==false时有效
	FeatureDimension int

	// 是否是监督式学习数据
	IsSupervisedLearning bool

	// 输出标注数（既分类数目）
	// 合法的标注值范围为[0, NumLabels-1]
	NumLabels int

	// 其它自定义的选项
	Options interface{}
}
```

可以通过数据集的GetOptions()函数得到该数据集的选项。

## 内存存储数据集

内存存储数据集（[inmem_dataset.go](/data/inmem_dataset.go)）将所有数据保存在内存中，因此是访问速度最快的一种数据集，如果你的数据可以完全载入内存，建议使用这种数据集。

使用方法如下：

一、在遍历数据前必须首先添加数据，例如

```go
set := data.NewInmemDataset()
if !set.AddInstance(instance1) {  // 添加另一个样本
  // 处理错误
}                                 // 反复调用AddInstance可以添加多条样本
set.Finalize()                    // 在所有数据添加完毕后必须调用此函数冻结数据
```

二、添加数据并冻结后可以用如下方式遍历数据。

```go
iter := set.CreateIterator()
iter.Start()
for !iter.End() {
  instance := iter.GetInstance()
  // 使用instance
  iter.Next()
}
```

三、在遍历数据的任何时刻可以使用Start()函数终止当前遍历开始新的遍历。

特别注意的是，AddInstance函数将会拥有传入的instance指针，所以请勿修改其内容。

## 跳跃数据集

跳跃数据集（[skip_dataset.go](/data/skip_dataset.go)）是一种建立在已有数据集上的数据集，其本身不创建任何新的数据。跳跃数据集存在的目的是为了能够跳跃式访问寄主数据集的部分数据，这对数据分割（data partition）很有用，比如在做模型的交叉评价（cross-validation）时。

跳跃数据集可以通过如下函数建立：

```go
func NewSkipDataset(set Dataset, buckets []SkipBucket) *skipDataset
```

buckets定义了怎样跳过数据集中的数据。SkipBucket结构体定义如下：

```go
type SkipBucket struct {
        SkipMode     bool
        NumInstances int
}
```

* 当SkipMode为true时，跳过NumInstances个数据
* 当SkipMode为false时，使用NumInstances个数据
* 重复buckets中的所有项，直到set遍历完为止。

和内存存储数据集一样，跳跃数据集中的数据访问需要通过CreateIterator()函数建立的遍历器进行遍历。

## 数据样本

数据集遍历器的GetInstance可以得到当前指向的数据样本，数据样本的格式如下：

```go
// 一条数据样本
//
// 对于监督式学习，数据样本通常包含了输入的特征值(features)和输出的
// 目标函数值（回归问题）或者标注（分类问题）。
// 对非监督式学习问题，之需要输入的特征值。
type Instance struct {
	// 输入的特征
	Features *util.Vector

	// 另一种表达输入特征的方式，以“特征名”：“特征值”的方式存放
	// 如果一个Instance仅有NamedFeatures，那么其中的特征会转化
	// 为稀疏矩阵存储在Features中。
	NamedFeatures map[string]float64

	// 输出
	// 仅当处理监督式学习问题时需要此项
	// 非监督式学习的数据请使用nil
	Output *InstanceOutput

	// 样本的字符串名，用以区分不同样本
	// 此项可为空
	Name string

	// 附加信息
	Attachment interface{}
}
```

***注意***：请仅仅使用Features和NamedFeatures两者之一来存储特征值，如果你两者都用，程序会优先使用NamedFeatures并自动转化为稀疏的Features，在这种情况下特征的ID可能是不确定的。

## 扩展

你可以添加新的数据集实现，比如当数据量很大无法载入一台机器内存时，可以从网络，数据库和文件系统中逐条或者批量载入到内存。只要自定义的数据集实现了Dataset和DatasetIterator的所有接口，就可以使用弥勒佛框架中的工具对其进行分析。
