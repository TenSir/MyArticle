# class Subgraph(nodes, relationships) 子图是节点和关系不可变的集合。

from py2neo import Node, Relationship

a = Node('Person', name='Alice')
b = Node('Person', name='Bob')
r = Relationship(a, 'KNOWS', b)
s = a | b | r

print(list(s))
for each in s:
    print(each)

print('r:',r)

# 还可以通过 nodes() 和 relationships() 方法获取所有的 Node 和 Relationship，实例如下：
print(s.nodes)
print(s.relationships)

for each in s.nodes:
    print(each)

for each in s.relationships:
    print(each)

# 另外还可以利用 & 取 Subgraph 的交集，例如：
print('____________________')
s1 = a | b | r
s2 = a | b
s3 = s1 & s2
print(s3)
for each in s3.nodes:
    print(each)


from py2neo import Node, Relationship
s = a | b | r
print(s.keys())
print(s.types())
print(len(s))



