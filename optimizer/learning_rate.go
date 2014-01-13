package optimizer

import (
	"github.com/huichen/mlf/util"
)

type LearningRate struct {
	rate               float64
	step               int
	characteristicTime float64
	oldDeltaX          *util.Matrix
}

func NewLearningRate(options OptimizerOptions) (lr *LearningRate) {
	lr = new(LearningRate)
	lr.rate = options.LearningRate
	lr.step = 0
	lr.characteristicTime = options.CharacteristicTime
	return
}

func (lr *LearningRate) ComputeLearningRate(newDeltaX *util.Matrix) (r float64) {
	if lr.characteristicTime == 0 {
		return lr.rate
	}
	r = lr.rate / (float64(1) + float64(lr.step)/float64(lr.characteristicTime))
	lr.step++
	return
}
