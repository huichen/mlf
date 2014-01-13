package data

// 数据集参数
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
