from py2neo import Graph, Node, Relationship

# 连接neo4j数据库，输入地址、用户名、密码
graph = Graph(
    "http://localhost:7474",
    username="neo4j",
    password='KG-neo4j'
)

graph.run("match(n) detach delete n")
a = Node('Person', name='Alice')
b = Node('Person', name='Bob')
r = Relationship(a, 'KNOWS', b)
s = a | b | r
graph.create(s)
# 使用data查询数据
# data = graph.run('MATCH (p:Person) return p').data
# print(data)

# 查看图结构中节点标签的类别，返回结果是一个frozenset
# 查看图结构中关系的类型
print(graph.schema.node_labels)
print(graph.schema.relationship_types)



cursor =graph.run('match (p:Person) return p.name limit 20')
print(cursor)

# record对象
"""
class py2neo.data.Record(iterable=)
	record[index]
	record[key]# 通过特定的键key或索引index返回值
	len(record)# 返回record的字段数
	dict(record)# 返回记录的字典格式
	data(*keys)# 以键值对的形式返回一个字典，只会选择*key或*index与其对应的值，如果*key中有key不在record中，依然会作为返回值，不过其值设为None;*index如果超出范围会报错。
	get(key,default=None)# 从record中按照键key或者索引index获取一个值，如果这个项不存在，则按默认值返回。
	index(key)# 返回键的index
	items(*keys)# 将记录的字段作为键列表和值元组返回，参数是项的键keys或indexs，如果为空，就全部包括,返回值是(key,value)元组的列表
	keys()# 返回记录的键keys
	to_subgraph()# 返回包含所有此记录中所有图形结构的union的子图。返回的是Subgraph对象。
	values(*keys)# 返回记录的值，根据索引或键选择过滤以只包含某些值。
"""
for each in cursor:
    print(each)
    print(each['p.name'])
print('___________________________________')

# 另外一个record的例子
gql="MATCH (p1:Person)-[k:KNOWS]->(p2:Person) RETURN *"
cursor=graph.run(gql)
# 循环向前移动游标
while cursor.forward():
    # 获取并打印当前的结果集
    record=cursor.current
    print('record:',record)

    record = cursor.current
    print('通过get返回：', record.get('k'))
    for (key, value) in record.items('p1', 'p2'):
        print('通过items返回元组：', key, ':', value)



###########cypher
print('___________________________________')
gql = "MATCH (p1:Person)-[k:KNOWS]->(p2:Person) RETURN *"
res = graph.run(gql)
print(res.data())

print(res.data())               # a list of dictionary
print(res.to_data_frame())      # pd.DataFrame
print(res.to_ndarray())         # numpy.ndarray
print(res.to_subgraph())
print(res.to_table())

