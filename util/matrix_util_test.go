package util

import (
	"testing"
)

func TestMatrixDotProduct(t *testing.T) {
	m1 := NewMatrix(2, 3)
	m1.GetValues(0).SetValues([]float64{1, 2, 3})
	m1.GetValues(1).SetValues([]float64{4, 5, 6})
	m2 := NewMatrix(2, 3)
	m2.GetValues(0).SetValues([]float64{7, 8, 9})
	m2.GetValues(1).SetValues([]float64{4, 5, 4})

	// 点乘积为 1*7+2*8+3*9+4*4+5*5+6*4 = 115
	Expect(t, "115", MatrixDotProduct(m1, m2))
}

func TestSparseMatrixDotProduct(t *testing.T) {
	m1 := NewSparseMatrix(2)
	m1.GetValues(0).SetValues([]float64{1, 2, 3})
	m1.GetValues(1).SetValues([]float64{4, 5, 6})
	m2 := NewSparseMatrix(2)
	m2.GetValues(0).SetValues([]float64{7, 8, 9})
	m2.GetValues(1).SetValues([]float64{4, 5, 4})

	// 点乘积为 1*7+2*8+3*9+4*4+5*5+6*4 = 115
	Expect(t, "115", MatrixDotProduct(m1, m2))
}
