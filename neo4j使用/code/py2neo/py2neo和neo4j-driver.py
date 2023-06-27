from py2neo import Subgraph

from py2neo import *

if __name__ == '__main__':
    graph = Graph(
        "http://127.0.0.1:7474",
        username="neo4j",
        password="KG-neo4j")

    graph.delete_all()

    node0 = Node('Person', name='Alice')
    node1 = Node('Person', name='Bob')
    node2 = Node('Person', name='Jack')

    node0['age'] = 20
    node1['age'] = 25
    node2['age'] = 50
    node0_know_node1 = Relationship(node1, 'know', node0)
    node2_know_node1 = Relationship(node1, 'know', node2)

    graph.create(node0)
    graph.create(node1)
    graph.create(node0_know_node1)
    graph.create(node2_know_node1)

    matcher = RelationshipMatcher(graph)
    result = matcher.match({node1}, 'know')

    for x in result:
        for y in walk(x):
            if type(y) is Node and y['age'] < 25:
                print(y['name'])
