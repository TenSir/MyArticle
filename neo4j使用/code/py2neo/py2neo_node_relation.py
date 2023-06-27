# 创建节点和关系
from py2neo import Node, Relationship

a = Node("Person", name="Alice")
b = Node("Person", name="Bob")
r = Relationship(a, "KNOWS", b)
print(r)


# 修改属性
a['age'] = 20
b['age'] = 21
r['time'] = '2017/08/31'
print(a, b, r)

# 使用setdefault
a.setdefault('location', '北京')

# update()方法对属性批量更新,字典方式更新
data = {
    'name': 'Amy',
    'age': 21
}
a.update(data)


########方法和属性
"""
###############其中包含的节点属性有：
hash(node) 返回node的ID的哈希值
node[key] 返回node的属性值，没有此属性就返回None
node[key] = value 设定node的属性值
del node[key] 删除属性值，如果不存在此属性报KeyError
len(node) 返回node属性的数量
dict(node) 返回node所有的属性
walk(node) 返回一个生成器且只包含一个node
labels() 返回node的标签的集合
has_label(label) node是否有这个标签
add_label(label) 给node添加标签
remove_label(label) 删除node的标签
clear_labels() 清楚node的所有标签
update_labels(labels) 添加多个标签，注labels为可迭代的

###############其中连接的属性有：
hash(relationship) 返回一个关系的hash值
relationship[key] 返回关系的属性值
relationship[key] = value 设定关系的属性值
del relationship[key] 删除关系的属性值
len(relationship) 返回关系的属性值数目
dict(relationship) 以字典的形式返回关系的所有属性
walk(relationship) 返回一个生成器包含起始node、关系本身、终止node
type() 返回关系type
"""
