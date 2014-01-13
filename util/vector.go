package util

import (
	"log"
	"math"
)

// 向量分两种类型：
//
// 1. 稠密向量，元素的key从0开始到N结束，实际值保存在切片中
//    使用NewVector(length)开辟新的稠密向量
//    稠密向量的长度在新建向量时已经指定好，无法更改
//
// 2. 稀疏向量，元素的key可以是不连续的值，保存在map中
//    使用NewSparseVector()开辟新的稀疏向量
//    稀疏向量主要用于处理特征稀疏的超大规模机器学习问题
//
// 需要注意的是
//
// 1. 请不要混合使用稀疏和稠密向量
// 2. 和其他类型一样，向量是协程不安全的
//
// 请使用如下方法遍历向量中元素的值
//   for _, key := range(vector.Keys()) {
//     vector.Get(key)
//   }
type Vector struct {
	// 稠密向量使用此切片保存元素的值
	values []float64

	// 稀疏向量使用此map保存元素的值
	valueMap map[int]float64

	// 对稀疏向量，这里保存这所有非零的元素的索引（从0开始）
	// 对稠密向量，keys[i] == i
	keys []int

	// 向量是否是稀疏向量
	isSparse bool
}

// 构造长度（维度）为length的稠密向量
func NewVector(length int) *Vector {
	v := new(Vector)
	v.values = make([]float64, length)
	v.keys = make([]int, length)
	for i := 0; i < length; i++ {
		v.keys[i] = i
	}
	v.Clear()
	v.isSparse = false
	return v
}

// 构造稀疏向量
func NewSparseVector() *Vector {
	v := new(Vector)
	v.valueMap = make(map[int]float64)
	v.keys = make([]int, 0)
	v.Clear()
	v.isSparse = true
	return v
}

// 向量值清零
func (v *Vector) Clear() {
	if v.isSparse {
		v.valueMap = make(map[int]float64)
		v.keys = make([]int, 0)
	} else {
		for k := 0; k < len(v.values); k++ {
			v.values[k] = 0.0
		}
	}
}

// 复制元素的值
func (v *Vector) DeepCopy(that *Vector) {
	if !v.IsHomogeneous(that) {
		log.Fatal("无法对两个不同质的向量做深度复制")
	}
	if v.isSparse {
		v.valueMap = make(map[int]float64)
		v.keys = make([]int, len(that.keys))
		for i, k := range that.keys {
			v.keys[i] = k
		}
		for k, va := range that.valueMap {
			v.valueMap[k] = va
		}
	} else {
		for k := 0; k < len(v.values); k++ {
			v.values[k] = that.values[k]
		}
	}
}

// 得到向量中单个元素的值
// 如果index不存在或者越界，返回0
func (v *Vector) Get(index int) float64 {
	if v.isSparse {
		return v.valueMap[index]
	}
	if index >= len(v.keys) {
		return 0
	}
	return v.values[index]
}

// v = v + alpha * that
func (v *Vector) Increment(that *Vector, alpha float64) {
	if !v.IsHomogeneous(that) {
		log.Fatal("无法对两个不同质的向量做Increment操作")
	}
	if v.isSparse {
		for i, k := range that.valueMap {
			_, ok := v.valueMap[i]
			v.valueMap[i] += k * alpha
			if !ok {
				v.keys = append(v.keys, i)
			}
		}
	} else {
		values1 := v.values
		values2 := that.values
		for i, k := range values1 {
			values1[i] = k + values2[i]*alpha
		}
	}
}

// 返回两个向量是否同质，同质的两个向量稀疏类型相同，如果都是稠密矩阵则长度也需要相同
func (v *Vector) IsHomogeneous(that *Vector) bool {
	if v.isSparse {
		if that.isSparse {
			return true
		} else {
			return false
		}
	} else {
		if that.isSparse {
			return false
		} else if len(v.keys) == len(that.keys) {
			return true
		}
	}
	return false
}

// 返回向量是否为稀疏向量
func (v *Vector) IsSparse() bool {
	return v.isSparse
}

// 返回向量索引的键值，用于遍历向量中的元素，使用方法见Vector结构体注释
func (v *Vector) Keys() []int {
	return v.keys
}

// v_i = (v_i*a + b) * that_i
func (v *Vector) Multiply(a, b float64, that *Vector) {
	if !v.IsHomogeneous(that) {
		log.Fatal("无法对两个不同质的向量做Multiply操作")
	}

	if v.isSparse {
		for i, k := range v.valueMap {
			v.valueMap[i] = (k*a + b) * that.valueMap[i]
		}
	} else {
		values := v.values
		valuesThat := that.values
		for i, k := range values {
			values[i] = (k*a + b) * valuesThat[i]
		}
	}
}

// 向量的2-模
func (v *Vector) Norm() float64 {
	var result float64
	if v.isSparse {
		for _, k := range v.keys {
			result += v.valueMap[k] * v.valueMap[k]
		}
	} else {
		for k := 0; k < len(v.values); k++ {
			result += v.values[k] * v.values[k]
		}
	}
	return math.Sqrt(result)
}

// 得到 -v
func (v *Vector) Opposite() *Vector {
	if v.isSparse {
		output := NewSparseVector()
		output.keys = make([]int, len(v.keys))
		for i, k := range v.keys {
			output.keys[i] = k
		}
		for k, va := range v.valueMap {
			output.valueMap[k] = -va
		}
		return output
	}
	output := NewVector(len(v.keys))
	for k := 0; k < len(v.values); k++ {
		output.values[k] = -v.values[k]
	}
	return output
}

// 返回一个空向量，此向量的类型（是否稀疏）和维度和v相同
func (v *Vector) Populate() *Vector {
	var r *Vector
	if v.isSparse {
		r = NewSparseVector()
	} else {
		r = NewVector(len(v.keys))
	}
	return r
}

// 设置向量中单个元素的值
func (v *Vector) Set(index int, value float64) {
	if v.isSparse {
		_, ok := v.valueMap[index]
		v.valueMap[index] = value
		if !ok {
			v.keys = append(v.keys, index)
		}
	} else {
		v.values[index] = value
	}
}

// 设置向量中所有元素的值为value
func (v *Vector) SetAll(value float64) {
	if v.isSparse {
		for i, _ := range v.valueMap {
			v.valueMap[i] = value
		}
	} else {
		values := v.values
		for i, _ := range v.values {
			values[i] = value
		}
	}
}

// 更新 v = s * v
func (v *Vector) Scale(s float64) {
	if v.isSparse {
		for _, k := range v.keys {
			v.valueMap[k] *= s
		}
	} else {
		for i, k := range v.values {
			v.values[i] = s * k
		}
	}
}

// 设置向量中多个元素的值，第i个元素设为为values[i]
func (v *Vector) SetValues(values []float64) {
	if v.isSparse {
		v.keys = make([]int, len(values))
		for i, va := range values {
			v.keys[i] = i
			v.valueMap[i] = va
		}
	} else {
		if len(v.keys) != len(values) {
			log.Fatal("SetValues参数切片长度和向量长度不一致")
		}
		for k := 0; k < len(v.values); k++ {
			v.values[k] = values[k]
		}
	}
}

// 计算两个向量的线性求和 v = a * Vector1 + b * Vector2
func (v *Vector) WeightedSum(Vector1, Vector2 *Vector, a, b float64) {
	if !v.IsHomogeneous(Vector1) || !v.IsHomogeneous(Vector2) {
		log.Fatal("无法对两个不同质的向量做WeightedSum操作")
	}

	if v.isSparse {
		if v == Vector1 || v == Vector2 {
			log.Fatal("WeightedSum参数不能为向量自己")
		}
		v.valueMap = make(map[int]float64)
		v.keys = make([]int, len(Vector1.keys))
		for i, k := range Vector1.Keys() {
			v.keys[i] = k
			v.valueMap[k] = a * Vector1.Get(i)
		}
		for i, k := range Vector2.Keys() {
			va, ok := v.valueMap[k]
			if ok {
				v.valueMap[k] = va + b*Vector2.Get(i)
			} else {
				v.keys = append(v.keys, k)
				v.valueMap[k] = b * Vector2.Get(i)
			}
		}
	} else {
		for k := 0; k < len(v.values); k++ {
			v.Set(k, a*Vector1.values[k]+b*Vector2.values[k])
		}
	}
}
