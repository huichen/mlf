package util

import (
	"testing"
)

func TestCircularBuffer(t *testing.T) {
	buffer := NewCircularBuffer(5)
	buffer.Push(1)
	buffer.Push(2)
	Expect(t, "3", buffer.Sum())
	Expect(t, "2", buffer.NumValues())
	buffer.Push(3)
	Expect(t, "3", buffer.NumValues())
	buffer.Push(4)
	buffer.Push(5)
	Expect(t, "15", buffer.Sum())
	Expect(t, "5", buffer.NumValues())
	buffer.Push(6)
	Expect(t, "20", buffer.Sum())
	Expect(t, "5", buffer.NumValues())
	buffer.Push(7)
	buffer.Push(8)
	Expect(t, "30", buffer.Sum())
	Expect(t, "5", buffer.NumValues())
	buffer.Push(9)
	buffer.Push(10)
	Expect(t, "40", buffer.Sum())
	Expect(t, "5", buffer.NumValues())
}
