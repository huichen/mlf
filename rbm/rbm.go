package rbm

import (
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/util"
	"log"
	"math"
	"math/rand"
	"sync"
)

// Restricted Boltzmann Machine
type RBM struct {
	lock struct {
		sync.RWMutex
		weights *util.Matrix
	}
	options RBMOptions
}

func NewRBM(options RBMOptions) *RBM {
	return &RBM{
		options: options,
	}
}

func (rbm *RBM) GetOptions() RBMOptions {
	return rbm.options
}

func (rbm *RBM) Train(set data.Dataset) {
	featureDimension := set.GetOptions().FeatureDimension
	visibleDim := featureDimension
	hiddenDim := rbm.options.NumHiddenUnits + 1
	log.Printf("#visible = %d, #hidden = %d", featureDimension-1, hiddenDim-1)

	// 随机化 weights
	rbm.lock.Lock()
	rbm.lock.weights = util.NewMatrix(hiddenDim, visibleDim)
	oldWeights := util.NewMatrix(hiddenDim, visibleDim)
	batchDerivative := util.NewMatrix(hiddenDim, visibleDim)
	for i := 0; i < hiddenDim; i++ {
		for j := 0; j < visibleDim; j++ {
			value := (rand.Float64()*2 - 1) * 0.01
			rbm.lock.weights.Set(i, j, value)
		}
	}
	rbm.lock.Unlock()

	// 启动工作协程
	ch := make(chan *data.Instance, rbm.options.Worker)
	out := make(chan *util.Matrix, rbm.options.Worker)
	for iWorker := 0; iWorker < rbm.options.Worker; iWorker++ {
		go rbm.derivativeWorker(ch, out, visibleDim, hiddenDim)
	}

	iteration := 0
	delta := 1.0
	for (rbm.options.MaxIter == 0 || iteration < rbm.options.MaxIter) &&
		(rbm.options.Delta == 0 || delta > rbm.options.Delta) {
		iteration++

		go rbm.feeder(set, ch)
		iBatch := 0
		batchDerivative.Clear()
		numInstances := set.NumInstances()
		for it := 0; it < numInstances; it++ {
			// 乱序读入
			derivative := <-out
			batchDerivative.Increment(derivative, rbm.options.LearningRate)
			iBatch++

			if iBatch == rbm.options.BatchSize || it == numInstances-1 {
				rbm.lock.Lock()
				rbm.lock.weights.Increment(batchDerivative, 1.0)
				rbm.lock.Unlock()
				iBatch = 0
				batchDerivative.Clear()
			}
		}

		// 统计delta和|weight|
		rbm.lock.RLock()
		weightsNorm := rbm.lock.weights.Norm()
		batchDerivative.DeepCopy(rbm.lock.weights)
		batchDerivative.Increment(oldWeights, -1.0)
		derivativeNorm := batchDerivative.Norm()
		delta = derivativeNorm / weightsNorm
		log.Printf("iter = %d, delta = %f, |weight| = %f",
			iteration, delta, weightsNorm)
		oldWeights.DeepCopy(rbm.lock.weights)
		rbm.lock.RUnlock()
	}
}

func (rbm *RBM) logistic(v *util.Vector, index int, isRow bool) (output float64) {
	output = 0.0
	if isRow {
		output = util.VecDotProduct(v, rbm.lock.weights.GetValues(index))
	} else {
		for i := 0; i < rbm.lock.weights.NumLabels(); i++ {
			output += v.Get(i) * rbm.lock.weights.Get(i, index)
		}
	}
	output = 1.0 / (1 + math.Exp(-output))
	return
}

func (rbm *RBM) bernoulli(p float64) float64 {
	if rand.Float64() < p {
		return 1.0
	} else {
		return 0.0
	}
}

func (rbm *RBM) derivativeWorker(ch chan *data.Instance, out chan *util.Matrix,
	visibleDim int, hiddenDim int) {
	// 可见单元
	visibleUnits := util.NewVector(visibleDim)
	visibleUnits.Set(0, 1.0)

	// 不可见单元
	hiddenUnitsProb := util.NewVector(hiddenDim)
	hiddenUnitsBinary := util.NewVector(hiddenDim)
	hiddenUnitsProb.Set(0, 1.0)
	hiddenUnitsBinary.Set(0, 1.0)

	for {
		instance := <-ch
		derivative := util.NewMatrix(hiddenDim, visibleDim)

		// 设置 visible units 的初始值
		for j := 1; j < visibleDim; j++ {
			visibleUnits.Set(j, instance.Features.Get(j))
		}

		rbm.lock.RLock()

		// 更新 hidden units
		for i := 1; i < hiddenDim; i++ {
			prob := rbm.logistic(visibleUnits, i, true)
			hiddenUnitsProb.Set(i, prob)
			if rbm.options.UseBinaryHiddenUnits {
				hiddenUnitsBinary.Set(i, rbm.bernoulli(prob))
			}
		}
		// 计算 positive statistics
		for i := 0; i < hiddenDim; i++ {
			for j := 0; j < visibleDim; j++ {
				derivative.Set(i, j, visibleUnits.Get(j)*hiddenUnitsProb.Get(i))
			}
		}

		// 计算CD_n
		for nCD := 0; nCD < rbm.options.NumCD; nCD++ {
			for j := 1; j < visibleDim; j++ {
				var prob float64
				if rbm.options.UseBinaryHiddenUnits {
					prob = rbm.logistic(hiddenUnitsBinary, j, false)
				} else {
					prob = rbm.logistic(hiddenUnitsProb, j, false)
				}
				visibleUnits.Set(j, prob)
			}
			for i := 1; i < hiddenDim; i++ {
				prob := rbm.logistic(visibleUnits, i, true)
				hiddenUnitsProb.Set(i, prob)
				if rbm.options.UseBinaryHiddenUnits {
					hiddenUnitsBinary.Set(i, rbm.bernoulli(prob))
				}
			}
		}

		rbm.lock.RUnlock()

		// 计算 negative statistics
		for i := 0; i < hiddenDim; i++ {
			for j := 0; j < visibleDim; j++ {
				old := derivative.Get(i, j)
				derivative.Set(i, j, old-visibleUnits.Get(j)*hiddenUnitsProb.Get(i))
			}
		}

		out <- derivative
	}
}

func (rbm *RBM) feeder(set data.Dataset, ch chan *data.Instance) {
	iter := set.CreateIterator()
	iter.Start()
	for it := 0; it < set.NumInstances(); it++ {
		instance := iter.GetInstance()
		ch <- instance
		iter.Next()
	}
}

// 输入和输出都有 bias 项
func (rbm *RBM) SampleHidden(v *util.Vector, n int, binary bool) *util.Vector {
	rbm.lock.RLock()
	defer rbm.lock.RUnlock()
	hiddenDim := rbm.options.NumHiddenUnits + 1
	visibleDim := rbm.lock.weights.NumValues()

	hiddenUnits := util.NewVector(hiddenDim)
	visibleUnits := util.NewVector(visibleDim)
	hiddenUnits.Set(0, 1.0)
	visibleUnits.Set(0, 1.0)

	for j := 1; j < visibleDim; j++ {
		visibleUnits.Set(j, v.Get(j))
	}

	// 更新 hidden units
	for i := 1; i < hiddenDim; i++ {
		prob := rbm.logistic(visibleUnits, i, true)
		if binary {
			hiddenUnits.Set(i, rbm.bernoulli(prob))
		} else {
			hiddenUnits.Set(i, prob)
		}
	}

	// reconstruct n-1 次
	for nCD := 0; nCD < n; nCD++ {
		for j := 1; j < visibleDim; j++ {
			var prob float64
			prob = rbm.logistic(hiddenUnits, j, false)
			visibleUnits.Set(j, prob)
		}
		for i := 1; i < hiddenDim; i++ {
			prob := rbm.logistic(visibleUnits, i, true)
			if binary {
				hiddenUnits.Set(i, rbm.bernoulli(prob))
			} else {
				hiddenUnits.Set(i, prob)
			}
		}
	}

	return hiddenUnits
}
