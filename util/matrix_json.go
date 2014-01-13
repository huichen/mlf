package util

import (
	"encoding/json"
)

// Matrix结构体JSON串行化/反串行化临时存储结构体
type MatrixJSON struct {
	Values    []*Vector
	NumValues int
	IsSparse  bool
}

// 对Matrix结构体进行JSON串行化
func (m *Matrix) MarshalJSON() ([]byte, error) {
	return json.Marshal(MatrixJSON{
		Values:    m.values,
		NumValues: m.numValues,
		IsSparse:  m.isSparse,
	})
}

// 对Matrix结构体进行JSON反串行化
func (m *Matrix) UnmarshalJSON(b []byte) error {
	var jsonData MatrixJSON
	err := json.Unmarshal(b, &jsonData)
	if err != nil {
		return err
	}

	m.values = jsonData.Values
	m.isSparse = jsonData.IsSparse
	m.numValues = jsonData.NumValues
	return nil
}
