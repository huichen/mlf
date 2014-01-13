package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/online"
	"io"
	"log"
	"net/http"
	"os"
	"runtime"
)

var (
	config_file = flag.String("config", "", "SGD分类服务器配置文件")

	classifier               *online.OnlineSGDClassifier
	instanceCount            int
	modelSavingInstanceCount int
	config                   TrainerServerConfig
)

type TrainerServerConfig struct {
	Host                       string
	Port                       int
	LoadModelPath              string
	SaveModelPath              string
	ModelSavingEveryNInstances int

	Options online.OnlineSGDClassifierOptions
}

type TrainResponse struct {
	Status       int
	ErrorMessage string
}

func main() {
	flag.Parse()
	runtime.GOMAXPROCS(1)

	if *config_file == "" {
		log.Fatal("必须指定--config")
	}
	config = LoadServerConfig(*config_file)

	classifier = online.NewOnlineSGDClassifier(config.Options)
	if config.LoadModelPath != "" {
		classifier.LoadWeightsFromFile(config.LoadModelPath)
	}

	http.HandleFunc("/train", TrainRpc)
	log.Print("服务器启动 ", config.Host, ":", config.Port)
	http.ListenAndServe(fmt.Sprintf("%s:%d", config.Host, config.Port), nil)
}

func LoadServerConfig(path string) (config TrainerServerConfig) {
	// 最大文件长度10M
	data := make([]byte, 1024*1024*10)

	f, err := os.Open(path)
	defer f.Close()
	if err != nil {
		log.Fatal("无法打开", path, "文件")
	}

	count, errRead := f.Read(data)
	if errRead != nil {
		log.Fatal("无法读入", path, "文件")
	}

	errUnmarshal := json.Unmarshal(data[:count], &config)
	if errUnmarshal != nil {
		log.Fatal("无法解析", path, "文件，错误", errUnmarshal)
	}

	return
}

func TrainRpc(w http.ResponseWriter, req *http.Request) {
	decoder := json.NewDecoder(req.Body)
	var instance data.Instance
	err := decoder.Decode(&instance)

	response := TrainResponse{}
	if err != nil {
		response.Status = 1
		response.ErrorMessage = fmt.Sprint(err)
	} else {
		classifier.TrainOnOneInstance(&instance)
		instanceCount++
		modelSavingInstanceCount++
		if instanceCount == config.Options.NumInstancesForEvaluation {
			output := classifier.Evaluate()
			log.Printf("+/p/r/f1/a %% = %.2f %.2f %.2f %.2f %.2f",
				100*output.Metrics["positive"],
				100*output.Metrics["precision"],
				100*output.Metrics["recall"],
				100*output.Metrics["fscore"],
				100*output.Metrics["accuracy"])
			instanceCount = 0
		}
		if modelSavingInstanceCount == config.ModelSavingEveryNInstances {
			if config.SaveModelPath != "" {
				classifier.Write(config.SaveModelPath)
			}
			modelSavingInstanceCount = 0
		}
		response.Status = 0
	}
	w.Header().Set("Content-Type", "application/json")
	responseJson, _ := json.Marshal(&response)
	io.WriteString(w, string(responseJson))
}
