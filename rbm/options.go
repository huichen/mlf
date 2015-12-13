package rbm

type RBMOptions struct {
	NumHiddenUnits       int
	NumCD                int
	UseBinaryHiddenUnits bool
	Worker               int

	LearningRate float64
	BatchSize    int
	Delta        float64
	MaxIter      int
}
