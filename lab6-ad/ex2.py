import networkx as nx
import matplotlib.pyplot as plt
import random

regular_graph = nx.random_regular_graph(3, 100)

caveman_graph = nx.connected_caveman_graph(10, 20)

merged_graph = nx.union(regular_graph, caveman_graph, rename=('r-', 'c-'))

nodes = list(merged_graph.nodes())
for _ in range(50):
    u = random.choice(nodes)
    v = random.choice(nodes)
    if u != v and not merged_graph.has_edge(u, v):
        merged_graph.add_edge(u, v)

pos = nx.spring_layout(merged_graph)
nx.draw(merged_graph, pos, with_labels=False, node_size=50, node_color='blue')

def detect_clique_nodes(graph):
    return list(graph.nodes())[:10]

clique_nodes = detect_clique_nodes(merged_graph)

nx.draw_networkx_nodes(merged_graph, pos, nodelist=clique_nodes, node_color='red', node_size=100)

plt.show()