弥勒佛
=====

**让天下没有难做的大数据模型！**

现有的机器学习框架/软件包存在几个问题：

* 无法处理大数据：多数Python，Matlab和R写的训练框架适合处理规模小的样本，没有为大数据优化。
* 不容易整合到实际生产系统：standalone的程序无法作为library嵌入到大程序中。
* 模型单一：一个软件包往往只解决一个类型的问题（比如监督式或者非监督式）。
* 不容易扩展：设计时没有考虑可扩展性，难以添加新的模型和组件。
* 代码质量不高：代码缺乏规范，难读懂、难维护。

弥勒佛项目的诞生就是为了解决上面的问题，在框架设计上满足了下面几个需求：

* **处理大数据**：可随业务增长scale up，无论你的数据样本是1K还是1B规模，都可使用弥勒佛项目。
* **为实际生产**：模型的训练和使用都可以作为library或者service整合到在生产系统中。
* **丰富的模型**：容易尝试不同的模型，在监督、非监督和在线学习等模型间方便地切换。
* **高度可扩展**：容易添加新模型，方便地对新模型进行实验并迅速整合到生产系统中。
* **高度可读性**：代码规范，注释和文档尽可能详尽，适合初学者进行大数据模型的学习。

# 安装/更新

```
go get -u github.com/huichen/mlf
```

# 功能

下面是弥勒佛框架解决的问题类型，括号中的斜体代表尚未实现以及预计实现的时间

* 监督式学习：[最大熵分类模型](/doc/maxent.md)（max entropy classifier），决策树模型（decision tree based models，*2014 Q1*）
* 非监督式学习：聚类问题（k-means，*2014 Q1*）
* 在线学习：[在线梯度递降模型](/doc/online.md)（online stochastic gradient descent）
* 神经网络（*2014 Q2/3*）

项目实现了下面的组件

* 多种[数据集](/doc/dataset.md)（in-mem，skip）
* 多种[评价器](/doc/eval.md)（precision，recall，f-score，accuracy，confusion）和[交叉评价](/doc/cross_validate.md)（cross-validation）
* 多种[优化器](/doc/optimizer.md)：协程并发L-BFGS，梯度递降（batch, mini-batch, stochastic），[带退火的学习率](/doc/optimizer.md#学习率)（learning rate），[L1/L2正则化](/doc/optimizer.md#正则化)（regularization）
* [稀疏向量](/doc/sparse_vector.md)（sparse vector）以存储和表达上亿级别的特征
* [特征辞典](/doc/dictionary.md)（feature dictionary）在特征名和特征ID之间自动翻译


# 其它

* [项目名称来历](/doc/naming.md)
* [项目邮件列表](https://groups.google.com/forum/#!forum/mlf-users)
* [联系方式](/doc/feedback.md)
