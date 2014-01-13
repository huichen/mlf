package util

func MatrixDotProduct(one, two *Matrix) float64 {
	// 当其中一个矩阵为空时返回0
	if one == nil || two == nil {
		return 0.0
	}

	result := float64(0)
	for i := 0; i < one.NumLabels(); i++ {
		result += VecDotProduct(one.GetValues(i), two.GetValues(i))
	}
	return result
}
