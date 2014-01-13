package util

import (
	"encoding/json"
	"strconv"
)

// Matrix结构体JSON串行化/反串行化临时存储结构体
type VectorJSON struct {
	Values   []float64
	ValueMap map[string]float64
	Keys     []int
	IsSparse bool
}

// 对Vector结构体进行JSON串行化
func (v *Vector) MarshalJSON() ([]byte, error) {
	vmap := make(map[string]float64)
	if v.isSparse {
		for _, k := range v.Keys() {
			vmap[strconv.Itoa(k)] = v.Get(k)
		}
	}
	return json.Marshal(VectorJSON{
		Values:   v.values,
		ValueMap: vmap,
		Keys:     v.keys,
		IsSparse: v.isSparse,
	})
}

// 对Vector结构体进行JSON反串行化
func (v *Vector) UnmarshalJSON(b []byte) error {
	var jsonData VectorJSON
	err := json.Unmarshal(b, &jsonData)
	if err != nil {
		return err
	}

	v.isSparse = jsonData.IsSparse
	v.keys = jsonData.Keys
	v.values = jsonData.Values
	v.valueMap = make(map[int]float64)
	if jsonData.IsSparse {
		for _, k := range v.Keys() {
			key := strconv.Itoa(k)
			v.valueMap[k] = jsonData.ValueMap[key]
		}
	}

	return nil
}
