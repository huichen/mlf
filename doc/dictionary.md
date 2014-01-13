特征词典
====

机器学习训练样本包含多个特征，标识一个特征可以使用字符串，这种方式容易被人阅读。
另一种方式是使用唯一的整数ID，整数的特征ID容易被机器处理，同时也更为节省存储空间。
特征词典组件的目的是为了自动将容易阅读的字符串转化为紧致和唯一的整数ID，并负责从整数ID到字符串的翻译。

特征词典必须通过[dictionary/dictionary.go](/dictionary/dictionary.go)中的函数创建

```go
func NewDictionary(minID) *Dictionary
```

请通过下面两个函数在特征名和特征ID之间翻译：

```go
func (d *Dictionary) TranslateIdFromName(name string) int
func (d *Dictionary) GetNameFromId(id int) string
```

***需要特别注意的是***：特征词典创建的特征ID是从1开始的，因为0要预留给常数项（值恒为1）特征。
