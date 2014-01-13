package dictionary

import (
	"encoding/json"
	"github.com/huichen/mlf/util"
	"testing"
)

func TestDictionaryJSON(t *testing.T) {
	dict := NewDictionary(1)
	dict.GetIdFromName("f1")
	dict.GetIdFromName("f2")
	dict.GetIdFromName("f3")
	dict.GetIdFromName("f4")

	dictJson, _ := json.Marshal(dict)

	var newDict Dictionary
	json.Unmarshal(dictJson, &newDict)

	util.Expect(t, "1", newDict.GetIdFromName("f1"))
	util.Expect(t, "2", newDict.GetIdFromName("f2"))
	util.Expect(t, "3", newDict.GetIdFromName("f3"))
	util.Expect(t, "4", newDict.GetIdFromName("f4"))

	util.Expect(t, "f1", newDict.GetNameFromId(1))
	util.Expect(t, "f2", newDict.GetNameFromId(2))
	util.Expect(t, "f3", newDict.GetNameFromId(3))
	util.Expect(t, "f4", newDict.GetNameFromId(4))
}
