package supervised

import (
	"github.com/huichen/mlf/data"
)

// 训练得到的机器学习模型
type Model interface {
	// 返回模型类型，比如"maxent_classifier"
	GetModelType() string

	// 将模型写入文件
	Write(path string)

	// 预测样本的输出
	Predict(instance *data.Instance) data.InstanceOutput
}
