package contrib

import (
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/util"
	"io/ioutil"
	"log"
	"strconv"
	"strings"
)

func LoadLibSVMDataset(path string, usingSparseRepresentation bool) data.Dataset {
	log.Print("载入libsvm格式文件", path)

	content, err := ioutil.ReadFile(path)
	if err != nil {
		log.Fatalf("无法打开文件\"%v\"，错误提示：%v\n", path, err)
	}
	lines := strings.Split(string(content), "\n")

	minFeature := 10000
	maxFeature := 0

	labels := make(map[string]int)
	labelIndex := 0

	for _, l := range lines {
		if l == "" {
			continue
		}

		fields := strings.Split(l, " ")

		_, ok := labels[fields[0]]
		if !ok {
			labels[fields[0]] = labelIndex
			labelIndex++
		}

		for i := 1; i < len(fields); i++ {
			if fields[i] == "" {
				continue
			}
			fs := strings.Split(fields[i], ":")
			fid, _ := strconv.Atoi(fs[0])
			if fid > maxFeature {
				maxFeature = fid
			}
			if fid < minFeature {
				minFeature = fid
			}
		}
	}

	if minFeature == 0 || maxFeature < 2 {
		log.Fatal("文件输入格式不合法")
	}
	log.Printf("feature 数目 %d", maxFeature)
	log.Printf("label 数目 %d", len(labels))

	set := data.NewInmemDataset()

	for _, l := range lines {
		if l == "" {
			continue
		}
		fields := strings.Split(l, " ")

		instance := new(data.Instance)
		instance.Output = &data.InstanceOutput{
			Label:       labels[fields[0]],
			LabelString: fields[0],
		}
		if usingSparseRepresentation {
			instance.NamedFeatures = make(map[string]float64)
		} else {
			instance.Features = util.NewVector(maxFeature + 1)
		}

		// 常数项
		if !usingSparseRepresentation {
			instance.Features.Set(0, 1)
		}

		for i := 1; i < len(fields); i++ {
			if fields[i] == "" {
				continue
			}
			fs := strings.Split(fields[i], ":")
			fid, _ := strconv.Atoi(fs[0])
			value, _ := strconv.ParseFloat(fs[1], 64)
			if usingSparseRepresentation {
				instance.NamedFeatures[fs[0]] = value
			} else {
				instance.Features.Set(fid, value)
			}
		}
		set.AddInstance(instance)
	}

	set.Finalize()

	log.Print("载入数据样本数目 ", set.NumInstances())

	return set
}
