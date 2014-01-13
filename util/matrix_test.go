package util

import (
	"testing"
)

func TestMatrixDeepCopy(t *testing.T) {
	m1 := NewMatrix(2, 3)
	m1.GetValues(0).SetValues([]float64{1, 2, 3})
	m1.GetValues(1).SetValues([]float64{3, 4, 2})

	// shallow copy
	m2 := m1
	Expect(t, "2", m2.Get(0, 1))
	m1.Set(0, 1, 3)
	Expect(t, "3", m2.Get(0, 1))

	// deep copy
	m3 := NewMatrix(2, 3)
	m3.DeepCopy(m1)
	Expect(t, "3", m3.Get(0, 1))
	m1.Set(0, 1, 4)
	Expect(t, "3", m3.Get(0, 1))
}

func TestMatrixIncrement(t *testing.T) {
	m1 := NewMatrix(2, 3)
	m1.GetValues(0).SetValues([]float64{1, 2, 3})
	m1.GetValues(1).SetValues([]float64{3, 4, 2})
	m2 := NewMatrix(2, 3)
	m2.GetValues(0).SetValues([]float64{1, 2, 2})
	m2.GetValues(1).SetValues([]float64{3, 1, 2})

	m1.Increment(m2, 2)
	Expect(t, "6", m1.Get(0, 1))
	Expect(t, "6", m1.Get(1, 2))
	Expect(t, "7", m1.Get(0, 2))
}

func TestMatrixIsSparse(t *testing.T) {
	v1 := NewMatrix(3, 1)
	Expect(t, "false", v1.IsSparse())
	v2 := NewSparseMatrix(3)
	Expect(t, "true", v2.IsSparse())
}

func TestMatrixNorm(t *testing.T) {
	m := NewMatrix(2, 2)
	m.GetValues(0).SetValues([]float64{9, 12})
	m.GetValues(1).SetValues([]float64{12, 16})
	Expect(t, "25", m.Norm())
}

func TestMatrixOpposite(t *testing.T) {
	m := NewMatrix(2, 10)
	m.Set(1, 3, 7)
	Expect(t, "7", m.Get(1, 3))

	m1 := m.Opposite()
	Expect(t, "-7", m1.Get(1, 3))
}

func TestMatrixPopulate(t *testing.T) {
	v := NewMatrix(2, 3)
	v.GetValues(0).SetValues([]float64{1, 2, 3})
	v.GetValues(1).SetValues([]float64{1, 3, 2})

	v1 := v.Populate()
	Expect(t, "2", v1.NumLabels())
	Expect(t, "0", v1.Get(0, 1))
	Expect(t, "0", v1.Get(0, 1))
	Expect(t, "0", v1.Get(1, 2))
}

func TestMatrixWeightedSum(t *testing.T) {
	m1 := NewMatrix(1, 3)
	m1.GetValues(0).SetValues([]float64{1, 2, 3})
	m2 := NewMatrix(1, 3)
	m2.GetValues(0).SetValues([]float64{3, 4, 5})

	m1.WeightedSum(m1, m2, 3, 4)
	Expect(t, "15", m1.Get(0, 0))
	Expect(t, "22", m1.Get(0, 1))
	Expect(t, "29", m1.Get(0, 2))
}

/******************************************************************************
稀疏矩阵版本
******************************************************************************/

func TestSparseMatrixDeepCopy(t *testing.T) {
	m1 := NewSparseMatrix(2)
	m1.GetValues(0).SetValues([]float64{1, 2, 3})
	m1.GetValues(1).SetValues([]float64{3, 4, 2})

	// shallow copy
	m2 := m1
	Expect(t, "2", m2.Get(0, 1))
	m1.Set(0, 1, 3)
	Expect(t, "3", m2.Get(0, 1))

	// deep copy
	m3 := NewSparseMatrix(2)
	m3.DeepCopy(m1)
	Expect(t, "3", m3.Get(0, 1))
	m1.Set(0, 1, 4)
	Expect(t, "3", m3.Get(0, 1))
}

func TestSparseMatrixIncrement(t *testing.T) {
	m1 := NewSparseMatrix(2)
	m1.GetValues(0).SetValues([]float64{1, 2, 3})
	m1.GetValues(1).SetValues([]float64{3, 4, 2})
	m2 := NewSparseMatrix(2)
	m2.GetValues(0).SetValues([]float64{1, 2, 2})
	m2.GetValues(1).SetValues([]float64{3, 1, 2})

	m1.Increment(m2, 2)
	Expect(t, "6", m1.Get(0, 1))
	Expect(t, "6", m1.Get(1, 2))
	Expect(t, "7", m1.Get(0, 2))
}

func TestSparseMatrixNorm(t *testing.T) {
	m := NewSparseMatrix(2)
	m.GetValues(0).SetValues([]float64{9, 12})
	m.GetValues(1).SetValues([]float64{12, 16})
	Expect(t, "25", m.Norm())
}

func TestSparseMatrixOpposite(t *testing.T) {
	m := NewSparseMatrix(2)
	m.Set(1, 3, 7)
	Expect(t, "7", m.Get(1, 3))

	m1 := m.Opposite()
	Expect(t, "-7", m1.Get(1, 3))
}

func TestSparseMatrixPopulate(t *testing.T) {
	v := NewSparseMatrix(2)
	v.GetValues(0).SetValues([]float64{1, 2, 3})
	v.GetValues(1).SetValues([]float64{1, 3, 2})

	v1 := v.Populate()
	Expect(t, "2", v1.NumLabels())
	Expect(t, "0", v1.Get(0, 1))
	Expect(t, "0", v1.Get(0, 1))
	Expect(t, "0", v1.Get(1, 2))
}

func TestSparseMatrixWeightedSum(t *testing.T) {
	m1 := NewSparseMatrix(1)
	m1.GetValues(0).SetValues([]float64{1, 2, 3})
	m2 := NewSparseMatrix(1)
	m2.GetValues(0).SetValues([]float64{3, 4, 5})

	m3 := NewSparseMatrix(1)
	m3.WeightedSum(m1, m2, 3, 4)
	Expect(t, "15", m3.Get(0, 0))
	Expect(t, "22", m3.Get(0, 1))
	Expect(t, "29", m3.Get(0, 2))
}
