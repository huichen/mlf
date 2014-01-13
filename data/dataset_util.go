package data

import (
	"github.com/huichen/mlf/dictionary"
	"github.com/huichen/mlf/util"
)

// 将instance中的NamedFeatures域转化为Features域
// 如果instance.Features不为nil则不转化
func ConvertNamedFeatures(instance *Instance, dict *dictionary.Dictionary) {
	if instance.Features != nil {
		return
	}

	instance.Features = util.NewSparseVector()
	// 第0个feature始终是1
	instance.Features.Set(0, 1.0)

	for k, v := range instance.NamedFeatures {
		id := dict.GetIdFromName(k)
		instance.Features.Set(id, v)
	}
}
