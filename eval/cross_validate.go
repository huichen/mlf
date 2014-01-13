package eval

import (
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/supervised"
)

// 进行N-fold cross-validation，输出评价
func CrossValidate(trainer supervised.Trainer, set data.Dataset,
	evals *Evaluators, folds int) (output Evaluation) {
	output.Metrics = make(map[string]float64)
	for iFold := 0; iFold < folds; iFold++ {
		// 裂分训练数据
		trainBuckets := []data.SkipBucket{
			{false, iFold},
			{true, 1},
			{false, folds - 1 - iFold},
		}
		trainSet := data.NewSkipDataset(set, trainBuckets)

		// 裂分评价数据
		evalBuckets := []data.SkipBucket{
			{true, iFold},
			{false, 1},
			{true, folds - 1 - iFold},
		}
		evalSet := data.NewSkipDataset(set, evalBuckets)

		// 在训练数据上训练模型
		model := trainer.Train(trainSet)

		// 在评价数据上评价
		metrics := evals.Evaluate(model, evalSet)

		// 累加评价结果
		for m, v := range metrics.Metrics {
			output.Metrics[m] += v
		}
	}

	// 评价结果求平均
	for m := range output.Metrics {
		output.Metrics[m] /= float64(folds)
	}

	return
}
