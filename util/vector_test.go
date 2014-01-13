package util

import (
	"testing"
)

func TestDeepCopy(t *testing.T) {
	vec1 := NewVector(3)
	vec1.SetValues([]float64{1, 2, 3})

	// shallow copy
	vec2 := vec1
	Expect(t, "1", vec2.Get(0))
	vec1.Set(0, 3)
	Expect(t, "3", vec2.Get(0))

	// deep copy
	vec3 := NewVector(3)
	vec3.DeepCopy(vec1)
	Expect(t, "3", vec3.Get(0))
	vec1.Set(0, 4)
	Expect(t, "3", vec3.Get(0))
}

func TestIncrement(t *testing.T) {
	vec1 := NewVector(3)
	vec1.SetValues([]float64{1, 2, 3})
	vec2 := NewVector(3)
	vec2.SetValues([]float64{1, 7, 2})

	vec1.Increment(vec2, 2)
	Expect(t, "3", vec1.Get(0))
	Expect(t, "16", vec1.Get(1))
	Expect(t, "7", vec1.Get(2))
}

func TestIsHomogeneous(t *testing.T) {
	vec1 := NewVector(3)
	vec2 := NewVector(3)

	Expect(t, "true", vec1.IsHomogeneous(vec1))
	Expect(t, "true", vec1.IsHomogeneous(vec2))
	Expect(t, "true", vec2.IsHomogeneous(vec1))

	vec3 := NewSparseVector()
	vec4 := NewVector(4)
	Expect(t, "false", vec1.IsHomogeneous(vec3))
	Expect(t, "false", vec3.IsHomogeneous(vec1))
	Expect(t, "false", vec1.IsHomogeneous(vec4))
	Expect(t, "false", vec4.IsHomogeneous(vec1))
	Expect(t, "false", vec3.IsHomogeneous(vec4))
	Expect(t, "false", vec4.IsHomogeneous(vec3))
}

func TestIsSparse(t *testing.T) {
	v1 := NewVector(3)
	Expect(t, "false", v1.IsSparse())
	v2 := NewSparseVector()
	Expect(t, "true", v2.IsSparse())
}

func TestVectorMultiply(t *testing.T) {
	v1 := NewVector(2)
	v1.SetValues([]float64{3, 4})
	v2 := NewVector(2)
	v2.SetValues([]float64{5, 6})

	v1.Multiply(2, 1, v2)
	Expect(t, "35", v1.Get(0))
	Expect(t, "54", v1.Get(1))
}

func TestVectorNorm(t *testing.T) {
	vec := NewVector(2)
	vec.SetValues([]float64{3, 4})
	Expect(t, "5", vec.Norm())
}

func TestVectorOpposite(t *testing.T) {
	vec := NewVector(10)
	vec.Set(3, 7)
	Expect(t, "7", vec.Get(3))

	vec1 := vec.Opposite()
	Expect(t, "-7", vec1.Get(3))
}

func TestVectorPopulate(t *testing.T) {
	v := NewVector(3)
	v.SetValues([]float64{1, 2, 3})

	v1 := v.Populate()
	Expect(t, "3", len(v1.Keys()))
	Expect(t, "0", v1.Get(0))
	Expect(t, "0", v1.Get(1))
	Expect(t, "0", v1.Get(2))
}

func TestVectorSetAall(t *testing.T) {
	vec := NewVector(2)
	vec.SetAll(3)
	Expect(t, "3", vec.Get(0))
	Expect(t, "3", vec.Get(1))
}

func TestVectorSetAndGet(t *testing.T) {
	vec := NewVector(10)
	Expect(t, "0", vec.Get(3))

	vec.Set(3, 7)
	Expect(t, "7", vec.Get(3))
}

func TestWeightedSum(t *testing.T) {
	vec1 := NewVector(3)
	vec1.SetValues([]float64{1, 2, 3})
	vec2 := NewVector(3)
	vec2.SetValues([]float64{3, 4, 5})

	vec1.WeightedSum(vec1, vec2, 3, 4)
	Expect(t, "15", vec1.Get(0))
	Expect(t, "22", vec1.Get(1))
	Expect(t, "29", vec1.Get(2))
}

/******************************************************************************
稀疏向量版本
******************************************************************************/

func TestSparseClear(t *testing.T) {
	vec1 := NewSparseVector()
	vec1.SetValues([]float64{1, 2, 3})

	Expect(t, "1", vec1.Get(0))
	vec1.Clear()
	Expect(t, "0", vec1.Get(0))
	Expect(t, "0", len(vec1.Keys()))
}

func TestSparseDeepCopy(t *testing.T) {
	vec1 := NewSparseVector()
	vec1.SetValues([]float64{1, 2, 3})

	// shallow copy
	vec2 := vec1
	Expect(t, "1", vec2.Get(0))
	vec1.Set(0, 3)
	Expect(t, "3", vec2.Get(0))

	// deep copy
	vec3 := NewSparseVector()
	vec3.DeepCopy(vec1)
	Expect(t, "3", vec3.Get(0))
	vec1.Set(0, 4)
	Expect(t, "3", vec3.Get(0))
}

func TestSparseIncrement(t *testing.T) {
	vec1 := NewSparseVector()
	vec1.SetValues([]float64{1, 2, 3})
	vec2 := NewSparseVector()
	vec2.SetValues([]float64{1, 7, 2})
	vec2.Set(3, 10)

	vec1.Increment(vec2, 2)
	Expect(t, "3", vec1.Get(0))
	Expect(t, "16", vec1.Get(1))
	Expect(t, "7", vec1.Get(2))
	Expect(t, "20", vec1.Get(3))
	Expect(t, "4", len(vec1.Keys()))
}

func TestSparseVectorMultiply(t *testing.T) {
	v1 := NewSparseVector()
	v1.SetValues([]float64{3, 4})
	v2 := NewSparseVector()
	v2.SetValues([]float64{5, 6})

	v1.Multiply(2, 1, v2)
	Expect(t, "35", v1.Get(0))
	Expect(t, "54", v1.Get(1))
}

func TestSparseVectorNorm(t *testing.T) {
	vec := NewSparseVector()
	vec.SetValues([]float64{3, 4})
	Expect(t, "5", vec.Norm())
}

func TestSparseVectorOpposite(t *testing.T) {
	vec := NewSparseVector()
	vec.Set(3, 7)
	Expect(t, "7", vec.Get(3))

	vec1 := vec.Opposite()
	Expect(t, "-7", vec1.Get(3))
	Expect(t, "1", len(vec1.Keys()))
}

func TestSparseVectorPopulate(t *testing.T) {
	v := NewSparseVector()
	v.SetValues([]float64{1, 2, 3})

	v1 := v.Populate()
	Expect(t, "0", len(v1.Keys()))
	Expect(t, "0", v1.Get(0))
	Expect(t, "0", v1.Get(1))
	Expect(t, "0", v1.Get(2))
}

func TestSparseVectorSetAall(t *testing.T) {
	vec := NewSparseVector()
	vec.SetValues([]float64{1, 2})
	vec.SetAll(3)
	Expect(t, "3", vec.Get(0))
	Expect(t, "3", vec.Get(1))
}

func TestSparseVectorSetAndGet(t *testing.T) {
	vec := NewSparseVector()
	Expect(t, "0", vec.Get(3))
	Expect(t, "0", len(vec.Keys()))

	vec.Set(3, 7)
	Expect(t, "7", vec.Get(3))
	Expect(t, "1", len(vec.Keys()))
}

func TestSparseWeightedSum(t *testing.T) {
	vec1 := NewSparseVector()
	vec1.SetValues([]float64{1, 2, 3})
	vec2 := NewSparseVector()
	vec2.SetValues([]float64{3, 4, 5})

	vec3 := NewSparseVector()
	vec3.WeightedSum(vec1, vec2, 3, 4)
	Expect(t, "15", vec3.Get(0))
	Expect(t, "22", vec3.Get(1))
	Expect(t, "29", vec3.Get(2))
	Expect(t, "3", len(vec3.Keys()))
}
