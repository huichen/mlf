package contrib

import (
	"testing"
)

func TestLibsvmSaver(t *testing.T) {
	set := LoadLibSVMDataset("test.txt", true)

	SaveLibSVMDataset("save_test.txt", set)
}
