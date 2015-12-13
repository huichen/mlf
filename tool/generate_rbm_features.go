package main

import (
	"flag"
	"fmt"
	"github.com/huichen/mlf/contrib"
	"github.com/huichen/mlf/rbm"
	"github.com/huichen/mlf/util"
	"runtime"
)

var (
	// 数据输入
	libsvm_file = flag.String("input", "", "libsvm格式的数据文件，训练数据")
	model       = flag.String("model", "model.mlf", "模型文件")
	append      = flag.Bool("append", false, "是否在原有 feature 之后添加新特征")
	numCD       = flag.Int("cd", 1, "CD次数")
	useBinary   = flag.Bool("use_binary", false, "是否使用抽样隐藏单元")
)

func main() {
	flag.Parse()
	runtime.GOMAXPROCS(runtime.NumCPU())

	// 载入训练集
	set := contrib.LoadLibSVMDataset(*libsvm_file, false)

	// 创建训练器
	machine := rbm.LoadRBM(*model)

	visibleDim := set.GetOptions().FeatureDimension
	hiddenDim := machine.GetOptions().NumHiddenUnits + 1

	iter := set.CreateIterator()
	iter.Start()
	for !iter.End() {
		instance := iter.GetInstance()
		v := util.NewVector(visibleDim)

		content := fmt.Sprintf("%s", instance.Output.LabelString)

		for i := 0; i < visibleDim; i++ {
			value := instance.Features.Get(i)
			v.Set(i, value)
			if value != 0.0 && *append {
				content = fmt.Sprintf("%s %d:%d", content, i+1, int(value))
			}
		}

		h := machine.SampleHidden(v, *numCD, *useBinary)

		for i := 1; i < hiddenDim; i++ {
			value := h.Get(i)
			if value != 0.0 {
				if *append {
					if *useBinary {
						content = fmt.Sprintf("%s %d:%d", content, visibleDim+i-1, int(value))
					} else {
						content = fmt.Sprintf("%s %d:%.3f", content, visibleDim+i-1, value)
					}
				} else {
					if *useBinary {
						content = fmt.Sprintf("%s %d:%d", content, i, int(value))
					} else {
						content = fmt.Sprintf("%s %d:%.3f", content, i, value)
					}
				}
			}
		}

		fmt.Printf("%s\n", content)

		iter.Next()
	}
}
