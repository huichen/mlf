在线学习
====

在线学习是机器学习的一种模式，在这种学习模式中训练数据以串行的数据流进入训练器，训练器动态更新模型参数，无需重复遍历数据。在线学习适用的情况

* 海量数据无法载入一台机器内存，训练数据需要以流的方式导入训练器
* 点击预测系统，需要模型能根据输入数据动态调整以达到实时响应

对于点击预测在线模型，训练数据和预测数据量通常差别两个数量级以上，因此我们一般将训练和预测放在不同的服务器上完成。简化的在线学习框架如下：

![](https://raw.github.com/huichen/mlf/master/doc/online.png)

其中，用户的点击行为首先存储在log中，经过抽样和去噪后喂给训练器，训练器动态调整模型参数并定期将模型写入文件，预测器服务集群定期导入模型并对实时的用户行为做出预测。

弥勒佛中实现了一个实验性质的在线学习系统，代码存放在[online](/online/)目录中。

## 训练服务器

训练服务器的代码存储在[online/trainer_server/trainer_server.go](/online/trainer_server/trainer_server.go)中。训练服务器从TrainerServerConfig格式的配置文件（JSON格式）中读入配置，比如[online/trainer_server/config_server.json](/online/trainer_server/config_trainer.json)文件：

```json
{
  "Host" : "127.0.0.1",
  "Port" : 8080,
  "LoadModelPath" : "",
  "SaveModelPath" : "model.mlf",
  "ModelSavingEveryNInstances" : 10000,
  "Options" : {
    "NumLabels" : 2,
    "BatchSize" : 1,
    "NumInstancesForEvaluation" : 10000,
    "Optimizer" : {
      "LearningRate" : 1,
      "RegularizationFactor" : 1,
      "RegularizationScheme" : 2
    }
  }
}
```

其中
* 训练前可载入一个初始模型，路径由LoadModelPath指定。
* 训练得到的模型定期（每训练ModelSavingEveryNInstances个样本）存储到SaveModelPath指定的路径。
* Options指定了模型的一些特性，比如分类数目，评价的样本数等等，详情见[online/online_sgd_options.go](/online/online_sgd_options.go)。

运行如下命令启动训练服务器：

```bash
go run trainer_server.go --config config_trainer.json
```

服务器通过HTTP服务的方式接收[data.Instance](/data/instance.go)格式的训练样本，并将评价指标在终端显示出来，比如

```bash
2014/01/12 19:10:35 +/p/r/f1/a % = 21.10 76.78 59.34 66.94 84.00
2014/01/12 19:10:36 +/p/r/f1/a % = 19.30 73.06 59.75 65.73 85.30
2014/01/12 19:10:36 +/p/r/f1/a % = 19.50 76.41 57.98 65.93 84.60
2014/01/12 19:10:36 +/p/r/f1/a % = 19.20 74.48 57.43 64.85 84.50
2014/01/12 19:10:37 +/p/r/f1/a % = 17.50 72.00 57.01 63.64 85.60
```

分别显示的是预测为正分类的样本百分比，精度，召回率，F1和准确率。评估的方法是这样的，在一个新的学习样本进来后，在更新参数之前先用旧的模型对此样本的输出进行预测，和标注进行比较，收集最近NumInstancesForEvaluation个样本的结果计算评价指标。为了保存最近的样本预测结果，弥勒佛框架定义了一个环形缓存结构体，见[代码](/util/circular_buffer.go)。

## 预测服务器

预测服务器的代码非常简单，见[online/prediction_server/prediction_server.go](/online/prediction_server/prediction_server.go)。和训练服务器一样，预测服务器接收JSON格式的训练样本并返回JSON格式的预测结果。

启动预测服务器请使用命令行

```bash
go run prediction_server.go --host 127.0.0.1 --port 8888 --model model_file.mlf
```

其中model_file.mlf指向训练服务器输出的模型文件。

## 喂食训练样本

[online/client/sgd_feeder.go](/online/client/sgd_feeder.go)提供了一个给训练服务器喂食样本的命令行工具：

```bash
go run sgd_feeder.go --input ../../testdata/a1a --server localhost:8080
```

其中--server来自你启动训练服务器时指定的地址和端口。喂食程序从--input指定的文件中读入数据，其中数据样本使用"特征名":"特征值"格式来指定特征向量，在训练服务器中会自动转化为整数ID表示，详情请见[data/instance.go](/data/instance.go)中的NamedFeatures的注释。

该程序也可以给预测服务器喂食样本，只要启动程序时指定 --mode predict 即可。

## 扩展

这里实现的训练服务器的QPS可以满足绝大多数实际生产的需要，但一个完整的系统远远比这篇文章中讨论的复杂，比如

* log里的样本需要去噪和抽样才能喂给训练服务器。
* 训练服务器使用的是stochastic gradient descent收敛，有速度更快的方法。
* 用JSON格式来传输样本和存储模型虽然方便阅读，但不是最有效的方式。
* 可以有更复杂的在线模型评价方式。

虽然如此，弥勒佛框架中已经实现了在线学习最核心的功能，是个好的起点，你可以通过添加组建来完善这个系统为你的业务服务 --- 这种系统必然是高度定制的，这也就是为什么弥勒佛中仅仅实现了一个基本功能的原因。
