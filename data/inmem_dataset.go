package data

import (
	"github.com/huichen/mlf/dictionary"
	"log"
)

// 内存中保存的数据集
//
// 使用方法如下：
// 1. 在遍历数据前必须首先添加数据，例如
//
//    set := NewInmemDataset(options)
//    if !set.AddInstance(instance1) {  // 添加另一个样本
//                                      // 处理错误
//    }                                 // 反复调用AddInstance可以添加多条样本
//    set.Finalize()                    // 在所有数据添加完毕后必须调用此函数冻结数据
//
// 2. 添加数据并冻结后可以用如下方式遍历数据
//
//    iter := set.CreateIterator()
//    iter.Start()
//    while !iter.End() {
//        instance := iter.GetInstance()
//        // 使用instance
//
//        iter.Next()
//    }
//
// 3. 在遍历数据的任何时刻可以使用Start()函数终止当前遍历开始新的遍历
//
// 特别注意的是，AddInstance函数将会拥有传入的instance指针，所以请勿修改其内容
type inmemDataset struct {
	instances []*Instance

	// 是否所有数据样本都已经添加完毕
	finalized bool

	options DatasetOptions

	featureDict, labelDict       *dictionary.Dictionary
	useFeatureDict, useLabelDict bool
}

func NewInmemDataset() *inmemDataset {
	set := new(inmemDataset)
	return set
}

func (set *inmemDataset) NumInstances() int {
	set.CheckFinalized(true)
	return len(set.instances)
}

func (set *inmemDataset) CreateIterator() DatasetIterator {
	return &inmemDatasetIterator{set: set}
}

func (set *inmemDataset) GetFeatureDictionary() *dictionary.Dictionary {
	return set.featureDict
}

func (set *inmemDataset) GetLabelDictionary() *dictionary.Dictionary {
	return set.labelDict
}

func (set *inmemDataset) GetOptions() DatasetOptions {
	return set.options
}

/******************************************************************************
inmemDataset特有的函数
******************************************************************************/

// 向数据集中添加一个样本
// 成功添加则返回true，否则返回false
func (set *inmemDataset) AddInstance(instance *Instance) bool {
	set.CheckFinalized(false)

	// 添加第一条样本时确定数据集的一些性质
	if len(set.instances) == 0 {
		if instance.NamedFeatures != nil {
			set.useFeatureDict = true
			set.featureDict = dictionary.NewDictionary(1) // 特征ID从0开始
			ConvertNamedFeatures(instance, set.featureDict)
		}

		if instance.Features.IsSparse() {
			set.options.FeatureIsSparse = true
			set.options.FeatureDimension = 0
		} else {
			set.options.FeatureIsSparse = false
			set.options.FeatureDimension = len(instance.Features.Keys())
		}

		if instance.Output == nil {
			set.options.IsSupervisedLearning = false
		} else {
			set.options.IsSupervisedLearning = true
			if instance.Output.LabelString != "" {
				set.useLabelDict = true
				set.labelDict = dictionary.NewDictionary(0)
				instance.Output.Label =
					set.labelDict.GetIdFromName(instance.Output.LabelString)
			}
		}
	} else {
		// 否则检查后续数据样本类型是否一致
		if instance.NamedFeatures != nil {
			ConvertNamedFeatures(instance, set.featureDict)
			if !set.useFeatureDict {
				log.Print("数据集不使用特征词典而添加的样本使用NamedFeatures")
				return false
			}
		} else {
			if set.useFeatureDict {
				log.Print("数据集使用特征词典而添加的样本不使用NamedFeatures")
				return false
			}
		}

		if set.options.FeatureIsSparse {
			if !instance.Features.IsSparse() {
				log.Print("数据集使用稀疏特征而添加的样本不稀疏")
				return false
			}
		} else {
			if instance.Features.IsSparse() {
				log.Print("数据集使用稠密特征而添加的样本稀疏")
				return false
			}

			if set.options.FeatureDimension != len(instance.Features.Keys()) {
				log.Print("数据集特征数和添加样本的特征数不同")
				return false
			}
		}

		if instance.Output == nil {
			if set.options.IsSupervisedLearning {
				log.Print("数据集为监督式而添加样本为非监督式数据")
				return false
			}
		} else {
			if !set.options.IsSupervisedLearning {
				log.Print("数据集为非监督式而添加样本为监督式数据")
				return false
			}

			if instance.Output.LabelString != "" {
				if !set.useLabelDict {
					log.Print("数据集不使用标注词典而添加的样本使用LabelString")
					return false
				}
			} else {
				if set.useLabelDict {
					log.Print("数据集使用标注词典而添加的样本不使用LabelString")
					return false
				}
			}
		}
	}

	if set.options.IsSupervisedLearning {
		if instance.Output.LabelString != "" {
			instance.Output.Label =
				set.labelDict.GetIdFromName(instance.Output.LabelString)
		}

		if instance.Output.Label < 0 {
			log.Println("样本标注值不在合法范围")
			return false
		}

		if instance.Output.Label >= set.options.NumLabels {
			set.options.NumLabels = instance.Output.Label + 1
		}
	}

	set.instances = append(set.instances, instance)
	return true
}

func (set *inmemDataset) Finalize() {
	set.CheckFinalized(false)
	set.finalized = true
}

func (set *inmemDataset) CheckFinalized(stat bool) {
	if set.finalized != stat {
		if stat {
			log.Fatal("在遍历数据前必须调用Finalize函数冻结数据")
		} else {
			log.Fatal("冻结数据后不能再对数据集进行修改")
		}
	}
}
