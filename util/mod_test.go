package util

import (
	"testing"
)

func TestMod(t *testing.T) {
	Expect(t, "3", Mod(3, 4))
	Expect(t, "0", Mod(0, 4))
	Expect(t, "0", Mod(4, 4))
	Expect(t, "5", Mod(14, 9))
}
