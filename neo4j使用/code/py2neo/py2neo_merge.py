from py2neo import Graph, Node, Relationship

graph = Graph(
    "http://localhost:7474",
    username="neo4j",
    password='KG-neo4j'
)

graph.delete_all()

a = Node("Person", name="孙悟空", age=10000)
b = Node("Person", name="猪八戒", age=5900)
KNOWS = Relationship.type("KNOWS")
graph.merge(KNOWS(a, b), "Person", "name")

# 新增一个三方节点
c = Node("Company", name="ACME")
c.__primarylabel__ = "Company"
c.__primarykey__ = "name"
WORKS_FOR = Relationship.type("WORKS_FOR")
graph.merge(WORKS_FOR(a, c) | WORKS_FOR(b, c))

