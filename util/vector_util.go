package util

import (
	"log"
)

// 计算点乘积 Vector1^T * Vector2
func VecDotProduct(Vector1, Vector2 *Vector) float64 {
	if !Vector1.IsHomogeneous(Vector2) {
		log.Fatal("无法对两个不同质的向量做点乘积")
	}

	var result float64
	result = 0
	if Vector1.IsSparse() {
		for i, k := range Vector1.valueMap {
			result += k * Vector2.valueMap[i]
		}
	} else {
		values1 := Vector1.values
		values2 := Vector2.values
		for i, v := range values1 {
			result += v * values2[i]
		}
	}
	return result
}
