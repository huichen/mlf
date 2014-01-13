package contrib

import (
	"bufio"
	"fmt"
	"github.com/huichen/mlf/data"
	"log"
	"os"
	"strconv"
)

func SaveLibSVMDataset(path string, set data.Dataset) {
	log.Print("保存数据集到libsvm格式文件", path)

	f, err := os.Create(path)
	defer f.Close()
	if err != nil {
		log.Fatalf("无法打开文件\"%v\"，错误提示：%v\n", path, err)
	}
	w := bufio.NewWriter(f)
	defer w.Flush()

	iter := set.CreateIterator()
	iter.Start()
	for !iter.End() {
		instance := iter.GetInstance()
		if instance.Output.LabelString == "" {
			fmt.Fprintf(w, "%d ", instance.Output.Label)
		} else {
			fmt.Fprintf(w, "%s ", instance.Output.LabelString)
		}
		for _, k := range instance.Features.Keys() {
			// 跳过第0个特征，因为它始终是1
			if k == 0 {
				continue
			}

			if instance.Features.Get(k) != 0 {
				// libsvm格式的特征从1开始
				fmt.Fprintf(w, "%d:%s ", k, strconv.FormatFloat(instance.Features.Get(k), 'f', -1, 64))
			}
		}
		fmt.Fprint(w, "\n")
		iter.Next()
	}
}
