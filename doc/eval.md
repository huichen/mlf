分类模型评价
====

监督式模型评价器的interface如下

```go
type Evaluator interface {
        Evaluate(m supervised.Model, set data.Dataset) Evaluation
}
```

在此接口上我们实现了下面几种metric

* 精度（precision）、召回率（recall）和F指数（f-score），见[precision_recall.go](/eval/precision_recall.go)
* 准确度（accuracy），见[accuracy.go](/eval/accuracy.go)
* 混淆矩阵，见[confusion_matrix.go](/eval/confusion_matrix.go)

Evaluator的Evaluate函数返回的Evaluation结构体实际上是个从度量名到值的映射：

```go
type Evaluation struct {
        Metrics map[string]float64
}
```

比如精度度量的名字为"precision"，准确度度量名为"accuracy"，不同度量的具体名称见各个度量的源代码文件中的注释。
