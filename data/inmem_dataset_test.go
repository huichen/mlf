package data

import (
	"github.com/huichen/mlf/util"
	"testing"
)

func TestInMemDataset(t *testing.T) {
	set := NewInmemDataset()

	instance1 := new(Instance)
	instance1.Features = util.NewVector(3)
	instance1.Features.SetValues([]float64{1, 2, 3})
	instance1.Output = &InstanceOutput{Label: 1}
	util.Expect(t, "true", set.AddInstance(instance1))

	instance2 := new(Instance)
	instance2.Features = util.NewVector(3)
	instance2.Features.SetValues([]float64{3, 4, 5})
	instance2.Output = &InstanceOutput{Label: 2}
	util.Expect(t, "true", set.AddInstance(instance2))

	instance3 := new(Instance)
	instance3.Features = util.NewSparseVector()
	instance3.Features.SetValues([]float64{3, 4, 5})
	instance3.Output = &InstanceOutput{Label: 0}
	util.Expect(t, "false", set.AddInstance(instance3))

	instance4 := new(Instance)
	instance4.Features = util.NewVector(4)
	instance4.Features.SetValues([]float64{3, 4, 5, 6})
	instance4.Output = &InstanceOutput{Label: 4}
	util.Expect(t, "false", set.AddInstance(instance4))

	instance5 := new(Instance)
	instance5.Features = util.NewVector(3)
	instance5.Features.SetValues([]float64{3, 5, 5})
	util.Expect(t, "false", set.AddInstance(instance5))

	set.Finalize()

	// 检查数据集选项
	util.Expect(t, "false", set.GetOptions().FeatureIsSparse)
	util.Expect(t, "3", set.GetOptions().FeatureDimension)
	util.Expect(t, "true", set.GetOptions().IsSupervisedLearning)
	util.Expect(t, "3", set.GetOptions().NumLabels)

	util.Expect(t, "2", set.NumInstances())

	iter := set.CreateIterator()
	iter.Start()
	util.Expect(t, "false", iter.End())
	util.Expect(t, "1", iter.GetInstance().Features.Get(0))
	util.Expect(t, "2", iter.GetInstance().Features.Get(1))
	util.Expect(t, "3", iter.GetInstance().Features.Get(2))

	iter.Next()
	util.Expect(t, "false", iter.End())
	util.Expect(t, "3", iter.GetInstance().Features.Get(0))
	util.Expect(t, "4", iter.GetInstance().Features.Get(1))
	util.Expect(t, "5", iter.GetInstance().Features.Get(2))

	iter.Next()
	util.Expect(t, "true", iter.End())

	iter = set.CreateIterator()
	iter.Start()
	iter.Skip(2)
	util.Expect(t, "true", iter.End())

	util.Expect(t, "2", set.NumInstances())
}
