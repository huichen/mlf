package data

import (
	"github.com/huichen/mlf/util"
	"testing"
)

func TestSkipIterator(t *testing.T) {
	set := NewInmemDataset()

	instance1 := new(Instance)
	instance1.Features = util.NewVector(3)
	instance1.Features.SetValues([]float64{1, 2, 3})
	util.Expect(t, "true", set.AddInstance(instance1))

	instance2 := new(Instance)
	instance2.Features = util.NewVector(3)
	instance2.Features.SetValues([]float64{3, 4, 5})
	util.Expect(t, "true", set.AddInstance(instance2))

	instance3 := new(Instance)
	instance3.Features = util.NewVector(3)
	instance3.Features.SetValues([]float64{7, 8, 9})
	util.Expect(t, "true", set.AddInstance(instance3))

	instance4 := new(Instance)
	instance4.Features = util.NewVector(3)
	instance4.Features.SetValues([]float64{30, 40, 50})
	util.Expect(t, "true", set.AddInstance(instance4))

	instance5 := new(Instance)
	instance5.Features = util.NewVector(3)
	instance5.Features.SetValues([]float64{70, 80, 90})
	util.Expect(t, "true", set.AddInstance(instance5))

	instance6 := new(Instance)
	instance6.Features = util.NewVector(3)
	instance6.Features.SetValues([]float64{31, 41, 51})
	util.Expect(t, "true", set.AddInstance(instance6))

	set.Finalize()

	iter := NewSkipIterator(set, []int{0, 4})
	iter.Start()
	util.Expect(t, "false", iter.End())
	util.Expect(t, "1", iter.GetInstance().Features.Get(0))
	util.Expect(t, "2", iter.GetInstance().Features.Get(1))
	util.Expect(t, "3", iter.GetInstance().Features.Get(2))

	iter.Next()
	util.Expect(t, "false", iter.End())
	util.Expect(t, "70", iter.GetInstance().Features.Get(0))
	util.Expect(t, "80", iter.GetInstance().Features.Get(1))
	util.Expect(t, "90", iter.GetInstance().Features.Get(2))

	iter.Next()
	util.Expect(t, "true", iter.End())

	iter = NewSkipIterator(set, []int{3, 1})
	iter.Start()
	util.Expect(t, "false", iter.End())
	util.Expect(t, "30", iter.GetInstance().Features.Get(0))
	util.Expect(t, "40", iter.GetInstance().Features.Get(1))
	util.Expect(t, "50", iter.GetInstance().Features.Get(2))

	iter.Next()
	util.Expect(t, "true", iter.End())

	iter = NewSkipIterator(set, []int{1, 2, 1})
	iter.Start()
	util.Expect(t, "false", iter.End())
	util.Expect(t, "3", iter.GetInstance().Features.Get(0))
	util.Expect(t, "4", iter.GetInstance().Features.Get(1))
	util.Expect(t, "5", iter.GetInstance().Features.Get(2))

	iter.Next()
	util.Expect(t, "false", iter.End())
	util.Expect(t, "30", iter.GetInstance().Features.Get(0))
	util.Expect(t, "40", iter.GetInstance().Features.Get(1))
	util.Expect(t, "50", iter.GetInstance().Features.Get(2))

	iter.Next()
	util.Expect(t, "false", iter.End())
	util.Expect(t, "31", iter.GetInstance().Features.Get(0))
	util.Expect(t, "41", iter.GetInstance().Features.Get(1))
	util.Expect(t, "51", iter.GetInstance().Features.Get(2))

	iter.Next()
	util.Expect(t, "true", iter.End())
}
