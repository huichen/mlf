package contrib

import (
	"github.com/huichen/mlf/util"
	"testing"
)

func TestLibsvmLoader(t *testing.T) {
	set := LoadLibSVMDataset("test.txt", false)
	util.Expect(t, "10", set.NumInstances())
	util.Expect(t, "45", set.GetOptions().FeatureDimension)
	util.Expect(t, "2", set.GetOptions().NumLabels)
}

func TestSparseLibsvmLoader(t *testing.T) {
	set := LoadLibSVMDataset("test.txt", true)
	util.Expect(t, "10", set.NumInstances())
	util.Expect(t, "0", set.GetOptions().FeatureDimension)
	util.Expect(t, "2", set.GetOptions().NumLabels)
}
