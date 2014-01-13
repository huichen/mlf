package util

import (
	"log"
	"math"
)

// 矩阵，用以存储多个标注对应的权重值
//
// 矩阵根据存储不同分为稀疏矩阵和稠密矩阵，区别见vector.go文件中对Vector结构体的注释
//
// 请不要直接创建Matrix，而是通过NewMatrix函数（稠密矩阵）和NewSparseMatrix（稀疏矩阵）
// 函数进行创建。
type Matrix struct {
	// values中每一个元素对应一个标注的值向量
	values []*Vector

	// 标注值向量的数目（特征维度）
	numValues int

	// 是否为稀疏矩阵
	isSparse bool
}

// 创建一个稠密矩阵，该矩阵有numLabels个标注，每个标注有numValues个值
func NewMatrix(numLabels, numValues int) *Matrix {
	m := new(Matrix)
	m.values = make([]*Vector, numLabels)
	for iLabel := 0; iLabel < numLabels; iLabel++ {
		m.values[iLabel] = NewVector(numValues)
	}
	m.numValues = numValues
	m.isSparse = false
	return m
}

// 创建一个稀疏矩阵，该矩阵有numLabels个标注，每个标注有numValues个值
func NewSparseMatrix(numLabels int) *Matrix {
	m := new(Matrix)
	m.values = make([]*Vector, numLabels)
	for iLabel := 0; iLabel < numLabels; iLabel++ {
		m.values[iLabel] = NewSparseVector()
	}
	m.isSparse = true
	return m
}

// 清空矩阵中的值，此函数不改变矩阵的维度
func (m *Matrix) Clear() {
	for _, v := range m.values {
		v.Clear()
	}
}

// 深度拷贝that矩阵
func (m *Matrix) DeepCopy(that *Matrix) {
	if m.NumLabels() != that.NumLabels() {
		log.Fatal("无法对两个维度不一致的矩阵进行DeepCopy操作")
	}

	for i := 0; i < m.NumLabels(); i++ {
		m.GetValues(i).DeepCopy(that.GetValues(i))
	}
}

// 得到矩阵中第label个标注的第index个值
// 如果index越界或者该值不存在，则返回0
func (m *Matrix) Get(label, index int) float64 {
	return m.values[label].Get(index)
}

// 得到第label个标注的值向量
func (m *Matrix) GetValues(label int) *Vector {
	return m.values[label]
}

// m = m + alpha * that
func (m *Matrix) Increment(that *Matrix, alpha float64) {
	if m.NumLabels() != that.NumLabels() {
		log.Fatal("无法对两个维度不一致的矩阵进行Increment操作")
	}

	for i := 0; i < m.NumLabels(); i++ {
		m.GetValues(i).Increment(that.GetValues(i), alpha)
	}
}

// 返回矩阵是否为稀疏矩阵
func (m *Matrix) IsSparse() bool {
	return m.isSparse
}

// 返回矩阵的标注数目
func (m *Matrix) NumLabels() int {
	return len(m.values)
}

// 返回标注值向量的长度
// 注意只能调用稠密矩阵的NumValues函数，对稀疏矩阵调用此函数非法
func (m *Matrix) NumValues() int {
	if m.isSparse {
		log.Fatal("无法调用稀疏矩阵的NumValues函数")
	}
	return m.numValues
}

// 返回矩阵中所有元素的2-Norm
func (m *Matrix) Norm() float64 {
	result := float64(0)
	for i := 0; i < m.NumLabels(); i++ {
		for _, k := range m.GetValues(i).Keys() {
			result += m.GetValues(i).Get(k) * m.GetValues(i).Get(k)
		}
	}
	return math.Sqrt(result)
}

// 返回 -m
func (m *Matrix) Opposite() *Matrix {
	var output *Matrix
	if m.isSparse {
		output = NewSparseMatrix(m.NumLabels())
	} else {
		output = NewMatrix(m.NumLabels(), m.NumValues())
	}
	for i := 0; i < m.NumLabels(); i++ {
		output.values[i] = m.GetValues(i).Opposite()
	}
	return output
}

// 返回一个空矩阵，此矩阵的类型（是否稀疏）和维度和m相同
func (m *Matrix) Populate() *Matrix {
	var ma *Matrix
	if m.isSparse {
		ma = NewSparseMatrix(len(m.values))
	} else {
		dimension := 0
		if len(m.values) > 0 {
			dimension = len(m.values[0].values)
		}
		ma = NewMatrix(len(m.values), dimension)
		for i := 0; i < len(m.values); i++ {
			ma.values[i] = NewVector(dimension)
		}
	}
	return ma
}

// m = s * m
func (m *Matrix) Scale(s float64) {
	for i := 0; i < m.NumLabels(); i++ {
		m.values[i].Scale(s)
	}
}

// 设置m矩阵label标注向量的index元素的值为value
func (m *Matrix) Set(label, index int, value float64) {
	m.values[label].Set(index, value)
}

// m = a * m1 + b * m2
// 注意m（指针）不能等于m1或者m2
func (m *Matrix) WeightedSum(m1, m2 *Matrix, a, b float64) {
	for i := 0; i < m.NumLabels(); i++ {
		m.values[i].WeightedSum(m1.GetValues(i), m2.GetValues(i), a, b)
	}
}
