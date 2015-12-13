package main

import (
	"flag"
	"github.com/huichen/mlf/contrib"
	"github.com/huichen/mlf/eval"
	"github.com/huichen/mlf/optimizer"
	"github.com/huichen/mlf/supervised"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
)

var (
	// 数据输入
	libsvm_file = flag.String("input", "", "libsvm格式的数据文件，训练数据")
	test_file   = flag.String("test", "", "libsvm格式的数据文件，测试数据")

	// 模型输出
	model_file = flag.String("output", "model.mlf", "模型输出")

	// 机器学习参数
	opt                 = flag.String("optimizer", "lbfgs", "优化器")
	reg                 = flag.Int("regularization", 2, "正则化方法")
	reg_factor          = flag.Float64("reg_factor", float64(1), "正则化因子")
	learning_rate       = flag.Float64("learning_rate", float64(1), "学习率")
	characteristic_time = flag.Float64("characteristic_time", float64(0), "学习率特征时间")
	batch_size          = flag.Int("batch_size", 10,
		"梯度递降法的batch尺寸: 0为full batch, 1为stochastic, 其它值为mini batch")
	delta = flag.Float64("delta", 1e-4,
		"权重变化量和权重的比值(|dw|/|w|)小于此值时判定为收敛")
	max_iter = flag.Int("max_iter", 0, "优化器最多迭代多少次")
	folds    = flag.Int("folds", 0, "N-交叉评价，值为零时不交叉评价")

	// 性能测试输出
	cpuprofile = flag.String("cpuprofile", "", "处理器profile文件")
)

func main() {
	flag.Parse()
	runtime.GOMAXPROCS(runtime.NumCPU())

	// 载入训练集
	set := contrib.LoadLibSVMDataset(*libsvm_file, false)

	// 设置训练器参数
	trainerOptions := supervised.TrainerOptions{
		Optimizer: optimizer.OptimizerOptions{
			OptimizerName:         *opt,
			RegularizationScheme:  *reg,
			RegularizationFactor:  *reg_factor,
			LearningRate:          *learning_rate,
			CharacteristicTime:    *characteristic_time,
			ConvergingDeltaWeight: *delta,
			ConvergingSteps:       3,
			MaxIterations:         *max_iter,
			GDBatchSize:           *batch_size,
		}}

	// 创建训练器
	trainer := supervised.NewMaxEntClassifierTrainer(trainerOptions)

	// 打开处理器profile文件
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	// 进行交叉评价
	// evaluators := eval.NewEvaluators([]eval.Evaluator{&eval.PREvaluator{}, &eval.AccuracyEvaluator{}})
	evaluators := eval.NewEvaluators([]eval.Evaluator{&eval.AccuracyEvaluator{}})
	if *folds != 0 {
		result := eval.CrossValidate(trainer, set, evaluators, *folds)
		log.Print(*folds, "-folds 交叉评价：")
		// log.Printf("精度   =  %.2f %%", result.Metrics["precision"]*100)
		// log.Printf("召回率 =  %.2f %%", result.Metrics["recall"]*100)
		// log.Printf("F1     =  %.2f %%", result.Metrics["fscore"]*100)
		log.Printf("准确度 =  %.2f %%", result.Metrics["accuracy"]*100)
		return
	}

	// 在全部数据上训练模型
	model := trainer.Train(set)
	model.Write(*model_file)

	// 测试模型
	if *test_file != "" {
		// 载入测试集
		testSet := contrib.LoadLibSVMDataset(*test_file, false)

		// 在测试集上评价模型并输出结果
		result := evaluators.Evaluate(model, testSet)
		log.Print("测试数据集评价：")
		// log.Printf("精度   =  %.2f %%", result.Metrics["precision"]*100)
		// log.Printf("召回率 =  %.2f %%", result.Metrics["recall"]*100)
		// log.Printf("F1     =  %.2f %%", result.Metrics["fscore"]*100)
		log.Printf("准确度 =  %.2f %%", result.Metrics["accuracy"]*100)
		log.Printf("error =  %.2f %%", 100.0-result.Metrics["accuracy"]*100)
	}
}
