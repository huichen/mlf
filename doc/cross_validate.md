交叉评价
====

交叉评价通过[eval/cross_validate.go](/eval/cross_validate.go)中的函数实现

```go
// 进行N-fold cross-validation，输出评价
func CrossValidate(
  trainer supervised.Trainer,
  set data.Dataset,
  evals *Evaluators,
  folds int) (output Evaluation)
```

此函数会在[数据集](/doc/dataset.md)set上建立[跳跃数据集](/doc/dataset.md#跳跃数据集)，将set分为folds份，然后遍历这folds份数据：

* 选定第i份数据，对剩余folds-1份数据利用trainer建立模型
* 用建立的模型对第i份数据进行评价，评价器为evals（[评价器](/doc/eval.md)数组）
* 遍历i，得到folds份评价，然后求平均
