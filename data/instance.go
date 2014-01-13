package data

import (
	"github.com/huichen/mlf/util"
)

// 一条数据样本
//
// 对于监督式学习，数据样本通常包含了输入的特征值(features)和输出的目标函数值。
// 对非监督式学习问题，只需要输入的特征值。
type Instance struct {
	// 输入的特征向量
	// Features和NamedFeatures同时不为nil时优先使用NamedFeatures生成稀疏特征向量并覆盖Features。
	Features *util.Vector

	// 另一种表达输入特征的方式，以“特征名”：“特征值”的方式存放
	// Features和NamedFeatures同时不为nil时优先使用NamedFeatures生成稀疏特征向量并覆盖Features。
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
