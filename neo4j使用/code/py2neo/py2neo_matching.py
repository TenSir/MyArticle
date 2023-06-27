from py2neo import Node, Graph, Relationship
from py2neo import NodeMatcher
from py2neo.matching import RelationshipMatcher
import py2neo

print(py2neo.__version__)
# 2020.0.0

# 连接neo4j 数据库
graph = Graph(
    'http://localhost:7474',
    auth=('neo4j','KG-neo4j')
)

graph.delete_all()

a = Node("Person", name="孙悟空", age=10000)
b = Node("Person", name="猪八戒", age=5900)
KNOWS = Relationship.type("KNOWS")
graph.merge(KNOWS(a, b), "Person", "name")


# node的数量
print('node的数量:',len(graph.nodes))

# class py2neo.matching.NodeMatcher(graph)
# class py2neo.matching.NodeMatch(graph, labels=frozenset(), conditions=(), order_by=(), skip=None, limit=None)
# class py2neo.matching.RelationshipMatcher(graph)
# class py2neo.matching.RelationshipMatch(graph, nodes=None, r_type=None, conditions=(), order_by=(), skip=None, limit=None)

# 使用NodeMatcher和RelationMatcher查询

s = graph.nodes.match("Person",name="孙悟空").first()
print(s)
print(s['name'])

print('______________________________')
node_matcher = NodeMatcher(graph)
node_res_1 = node_matcher.match("Person")
node_res_2 = node_matcher.match("Person", name="孙悟空").first()
print('node_res_1:',node_res_1)
print('len(node_res_1):',len(node_res_1))
print('node_res_2:',node_res_2)
print(len(node_matcher.match("Person").where("_.name =~ '孙.*'")))
print(list(node_matcher.match("Person").where("_.name =~ '孙.*'", "1980 <= _.age < 12000")))
# 排序
print(list(node_matcher.match("Person").order_by('_.age')))


print('______________________________')
relation_matcher = RelationshipMatcher(graph)

# 查询前10条KNOWS的关系
print(list(relation_matcher.match(r_type="KNOWS").limit(10)))

result = node_matcher.match("Person", name="孙悟空").first()
ret = relation_matcher.match((result,), r_type="KNOWS").first()
print(ret)
