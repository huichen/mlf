package dictionary

import (
	"github.com/huichen/mlf/util"
	"testing"
)

func TestDictionary(t *testing.T) {
	dict := NewDictionary(1)
	util.Expect(t, "1", dict.GetIdFromName("feature1"))
	util.Expect(t, "2", dict.GetIdFromName("feature2"))
	util.Expect(t, "3", dict.GetIdFromName("feature3"))
	util.Expect(t, "2", dict.GetIdFromName("feature2"))
	util.Expect(t, "3", dict.GetIdFromName("feature3"))
	util.Expect(t, "1", dict.GetIdFromName("feature1"))
	util.Expect(t, "4", dict.GetIdFromName("feature4"))

	util.Expect(t, "", dict.GetNameFromId(0))
	util.Expect(t, "feature1", dict.GetNameFromId(1))
	util.Expect(t, "feature2", dict.GetNameFromId(2))
	util.Expect(t, "feature3", dict.GetNameFromId(3))
	util.Expect(t, "feature4", dict.GetNameFromId(4))
	util.Expect(t, "", dict.GetNameFromId(5))
}
