package util

// 计算 a mod b
func Mod(a, b int) int {
	for a >= b {
		a -= b
	}

	return a
}
