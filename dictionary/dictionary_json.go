package dictionary

import (
	"encoding/json"
)

// Dictionary结构体JSON串行化/反串行化临时存储结构体
type DictionaryJSON struct {
	Features map[string]int
}

// 对Dictionary结构体进行JSON串行化
func (m *Dictionary) MarshalJSON() ([]byte, error) {
	return json.Marshal(DictionaryJSON{
		Features: m.nameToId,
	})
}

// 对Dictionary结构体进行JSON反串行化
func (m *Dictionary) UnmarshalJSON(b []byte) error {
	var jsonData DictionaryJSON
	err := json.Unmarshal(b, &jsonData)
	if err != nil {
		return err
	}

	m.nameToId = jsonData.Features
	m.idToName = make(map[int]string)
	for k, v := range m.nameToId {
		m.idToName[v] = k
		if m.maxId < v {
			m.maxId = v
		}
	}
	return nil
}
