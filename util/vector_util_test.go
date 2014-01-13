package util

import (
	"testing"
)

func TestVecDotProduct(t *testing.T) {
	vec1 := NewVector(3)
	vec1.SetValues([]float64{1, 2, 3})
	vec2 := NewVector(3)
	vec2.SetValues([]float64{3, 4, 5})

	// 点乘积为 1*3+2*4+3*5 = 26
	Expect(t, "26", VecDotProduct(vec1, vec2))
}

func TestSparseVecDotProduct(t *testing.T) {
	vec1 := NewSparseVector()
	vec1.SetValues([]float64{1, 2, 3})
	vec2 := NewSparseVector()
	vec2.SetValues([]float64{3, 4, 5})

	// 点乘积为 1*3+2*4+3*5 = 26
	Expect(t, "26", VecDotProduct(vec1, vec2))
}
