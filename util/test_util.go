package util

import (
	"fmt"
	"math"
	"testing"
)

func Expect(t *testing.T, expect string, actual interface{}) {
	actualString := fmt.Sprint(actual)
	if expect != actualString {
		t.Errorf("期待值=\"%s\", 实际=\"%s\"", expect, actualString)
	}
}

func ExpectNear(t *testing.T, expect float64, actual float64, acc float64) {
	if math.Abs(expect-actual) > acc {
		t.Errorf("期待值=\"%s\", 实际=\"%s\"", expect, actual)
	}
}
