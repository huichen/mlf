package dictionary

// 名称翻译词典
// 负责在名称和整数ID之间互相翻译
// 注意名称不能为空，ID从1开始
type Dictionary struct {
	nameToId map[string]int
	idToName map[int]string
	maxId    int
	minId    int
}

// 新建词典
// 必须通过这个函数创建新词典
func NewDictionary(minId int) *Dictionary {
	dict := new(Dictionary)
	dict.nameToId = make(map[string]int)
	dict.idToName = make(map[int]string)
	dict.minId = minId
	dict.maxId = minId
	return dict
}

// 从名称得到整数ID
// 对从未见过的，为其创建新ID并返回，否则直接返回已有ID
func (d *Dictionary) GetIdFromName(name string) int {
	id, ok := d.nameToId[name]
	if ok {
		return id
	}

	d.nameToId[name] = d.maxId
	d.idToName[d.maxId] = name
	d.maxId++
	return d.maxId - 1
}

// 从ID得到名称
// 如果ID不存在则返回空字符串
func (d *Dictionary) GetNameFromId(id int) string {
	return d.idToName[id]
}

// 从名称得到整数ID
// 对从未见过的，返回-1，否则直接返回已有ID
func (d *Dictionary) TranslateIdFromName(name string) int {
	id, ok := d.nameToId[name]
	if ok {
		return id
	}
	return -1
}
