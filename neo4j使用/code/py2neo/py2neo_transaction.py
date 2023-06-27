from py2neo import Graph, Relationship


from py2neo import Graph, Node, Relationship
graph = Graph(
    "http://localhost:7474",
    username="neo4j",
    password='KG-neo4j'
)

print(graph.delete_all())


tx = graph.begin()
a = Node("Person", name="Alice")
tx.create(a)
b = Node("Person", name="Bob")
ab = Relationship(a, "KNOWS", b)
tx.create(ab)
tx.commit()
print(graph.exists(ab))


c = Node("Person", name="Carol")
class WorksWith(Relationship):
    pass
ac = WorksWith(a, c)
print(type(ac))

"""
py2neo
driver可能触发多次client端和server端的通信，复杂查询延迟较大，适用如下场合：
1）简单的cypher查询，如单个顶点查询、一跳关联查询
2）期望返回数据为node类型的多跳复杂查询，不关注查询延迟，数据可以多次分批获得

neo4j restful
api查询在client和server端仅需要一次通信，适用场合如下：
1）对延迟敏感的多跳复杂查询，期望一次拿到所有数据
"""

