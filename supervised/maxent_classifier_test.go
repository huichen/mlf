package supervised

import (
	"github.com/huichen/mlf/data"
	"github.com/huichen/mlf/optimizer"
	"github.com/huichen/mlf/util"
	"testing"
)

func TestComputeInstanceDerivative(t *testing.T) {
	weights := util.NewMatrix(2, 3)
	weights.GetValues(0).SetValues([]float64{1, 2, 3})
	weights.GetValues(1).SetValues([]float64{3, 4, 5})
	de := util.NewMatrix(2, 3)
	instance := data.Instance{}
	instance.Features = util.NewVector(3)

	instance.Features.SetValues([]float64{1, 0.3, 0.4})
	instance.Output = &data.InstanceOutput{Label: 0}
	MaxEntComputeInstanceDerivative(weights, &instance, de)

	util.ExpectNear(t, 0.0322, de.Get(0, 0), 0.0001)
	util.ExpectNear(t, 0.0096, de.Get(0, 1), 0.0001)
	util.ExpectNear(t, 0.0128, de.Get(0, 2), 0.0001)
	util.ExpectNear(t, 0.9658, de.Get(1, 0), 0.0001)
	util.ExpectNear(t, 0.2897, de.Get(1, 1), 0.0001)
	util.ExpectNear(t, 0.3863, de.Get(1, 2), 0.0001)

	instance.Features.SetValues([]float64{1, 0.6, 0.7})
	instance.Output.Label = 1
	MaxEntComputeInstanceDerivative(weights, &instance, de)
	util.ExpectNear(t, -0.9900, de.Get(0, 0), 0.0001)
	util.ExpectNear(t, -0.5940, de.Get(0, 1), 0.0001)
	util.ExpectNear(t, -0.6930, de.Get(0, 2), 0.0001)
	util.ExpectNear(t, 0.9899, de.Get(1, 0), 0.0001)
	util.ExpectNear(t, 0.5939, de.Get(1, 1), 0.0001)
	util.ExpectNear(t, 0.6929, de.Get(1, 2), 0.0001)

	instance.Features.SetValues([]float64{1, 0.4, 0.2})
	instance.Output.Label = 2
	MaxEntComputeInstanceDerivative(weights, &instance, de)
	util.ExpectNear(t, 0.0390, de.Get(0, 0), 0.0001)
	util.ExpectNear(t, 0.0156, de.Get(0, 1), 0.0001)
	util.ExpectNear(t, 0.0078, de.Get(0, 2), 0.0001)
	util.ExpectNear(t, -0.0425, de.Get(1, 0), 0.0001)
	util.ExpectNear(t, -0.0170, de.Get(1, 1), 0.0001)
	util.ExpectNear(t, -0.0085, de.Get(1, 2), 0.0001)
}

func TestTrain(t *testing.T) {
	set := data.NewInmemDataset()
	instance1 := new(data.Instance)
	instance1.Features = util.NewVector(4)
	instance1.Features.SetValues([]float64{1, 1, 1, 3})
	instance1.Output = &data.InstanceOutput{Label: 0}
	set.AddInstance(instance1)

	instance2 := new(data.Instance)
	instance2.Features = util.NewVector(4)
	instance2.Features.SetValues([]float64{1, 3, 1, 5})
	instance2.Output = &data.InstanceOutput{Label: 0}
	set.AddInstance(instance2)

	instance3 := new(data.Instance)
	instance3.Features = util.NewVector(4)
	instance3.Features.SetValues([]float64{1, 3, 4, 7})
	instance3.Output = &data.InstanceOutput{Label: 1}
	set.AddInstance(instance3)

	instance4 := new(data.Instance)
	instance4.Features = util.NewVector(4)
	instance4.Features.SetValues([]float64{1, 2, 8, 6})
	instance4.Output = &data.InstanceOutput{Label: 1}
	set.AddInstance(instance4)
	set.Finalize()

	gdTrainerOptions := TrainerOptions{
		Optimizer: optimizer.OptimizerOptions{
			OptimizerName:         "gd",
			RegularizationScheme:  2,
			RegularizationFactor:  1,
			LearningRate:          0.1,
			ConvergingDeltaWeight: 1e-6,
			ConvergingSteps:       3,
			MaxIterations:         0,
			GDBatchSize:           0, // full-bath
		},
	}
	gdTrainer := NewMaxEntClassifierTrainer(gdTrainerOptions)

	lbfgsTrainerOptions := TrainerOptions{
		Optimizer: optimizer.OptimizerOptions{
			OptimizerName:         "lbfgs",
			RegularizationScheme:  2,
			RegularizationFactor:  1,
			LearningRate:          1,
			ConvergingDeltaWeight: 1e-6,
			ConvergingSteps:       3,
			MaxIterations:         0,
		},
	}
	lbfgsTrainer := NewMaxEntClassifierTrainer(lbfgsTrainerOptions)
	lbfgsTrainer.Train(set)

	gdTrainer.Train(set).Write("test.mlf")
	model := LoadModel("test.mlf")
	util.Expect(t, "0", model.Predict(instance1).Label)
	util.Expect(t, "0", model.Predict(instance2).Label)
	util.Expect(t, "1", model.Predict(instance3).Label)
	util.Expect(t, "1", model.Predict(instance4).Label)
}

func TestTrainWithNamedFeatures(t *testing.T) {
	set := data.NewInmemDataset()
	instance1 := new(data.Instance)
	instance1.NamedFeatures = map[string]float64{
		"1": 1,
		"2": 1,
		"3": 1,
		"4": 3,
	}
	instance1.Output = &data.InstanceOutput{Label: 0}
	set.AddInstance(instance1)

	instance2 := new(data.Instance)
	instance2.NamedFeatures = map[string]float64{
		"1": 1,
		"2": 3,
		"3": 1,
		"4": 5,
	}
	instance2.Output = &data.InstanceOutput{Label: 0}
	set.AddInstance(instance2)

	instance3 := new(data.Instance)
	instance3.NamedFeatures = map[string]float64{
		"1": 1,
		"2": 3,
		"3": 4,
		"4": 7,
	}
	instance3.Output = &data.InstanceOutput{Label: 1}
	set.AddInstance(instance3)

	instance4 := new(data.Instance)
	instance4.NamedFeatures = map[string]float64{
		"1": 1,
		"2": 2,
		"3": 8,
		"4": 6,
	}
	instance4.Output = &data.InstanceOutput{Label: 1}
	set.AddInstance(instance4)
	set.Finalize()

	gdTrainerOptions := TrainerOptions{
		Optimizer: optimizer.OptimizerOptions{
			OptimizerName:         "gd",
			RegularizationScheme:  2,
			RegularizationFactor:  1,
			LearningRate:          0.1,
			ConvergingDeltaWeight: 1e-6,
			ConvergingSteps:       3,
			MaxIterations:         0,
			GDBatchSize:           0, // full-bath
		},
	}
	gdTrainer := NewMaxEntClassifierTrainer(gdTrainerOptions)

	lbfgsTrainerOptions := TrainerOptions{
		Optimizer: optimizer.OptimizerOptions{
			OptimizerName:         "lbfgs",
			RegularizationScheme:  2,
			RegularizationFactor:  1,
			LearningRate:          1,
			ConvergingDeltaWeight: 1e-6,
			ConvergingSteps:       3,
			MaxIterations:         0,
		},
	}
	lbfgsTrainer := NewMaxEntClassifierTrainer(lbfgsTrainerOptions)
	lbfgsTrainer.Train(set)

	gdTrainer.Train(set).Write("test.mlf")
	model := LoadModel("test.mlf")
	util.Expect(t, "0", model.Predict(instance1).Label)
	util.Expect(t, "0", model.Predict(instance2).Label)
	util.Expect(t, "1", model.Predict(instance3).Label)
	util.Expect(t, "1", model.Predict(instance4).Label)
}
