import networkx as nx
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

def load_graph(file_path, max_lines=1500):
    G = nx.Graph()
    
    with open(file_path, 'r') as file:
        #skip first 4 lines
        file.readline()
        file.readline()
        file.readline()
        file.readline()
        
        for i, line in enumerate(file):
            if i >= max_lines:
                break
            node1, node2 = map(int, line.strip().split())
            if G.has_edge(node1, node2):
                G[node1][node2]['weight'] += 1
            else:
                G.add_edge(node1, node2, weight=1)
    
    return G

def extract_features(G):
    features = {}
    
    for node in G.nodes():
        egonet = G.subgraph(G.neighbors(node))
        
        Ni = len(egonet.nodes())
        Ei = len(egonet.edges())
        Wi = sum(data['weight'] for u, v, data in egonet.edges(data=True))
        
        adjacency_matrix = nx.to_numpy_array(egonet, weight='weight')
        eigenvalues = np.linalg.eigvals(adjacency_matrix)
        lambda_w_i = max(eigenvalues)
        
        features[node] = {
            'Ni': Ni,
            'Ei': Ei,
            'Wi': Wi,
            'lambda_w_i': lambda_w_i
        }
    
    nx.set_node_attributes(G, features)
    return features

def compute_anomaly_scores(G, features):
    nodes = list(G.nodes())
    Ei = np.array([features[node]['Ei'] for node in nodes])
    Ni = np.array([features[node]['Ni'] for node in nodes])
    
    log_Ei = np.log(Ei + 1)
    log_Ni = np.log(Ni + 1)
    
    X = log_Ni.reshape(-1, 1)
    y = log_Ei
    
    model = LinearRegression()
    model.fit(X, y)
    
    C = np.exp(model.intercept_)
    theta = model.coef_[0]
    
    anomaly_scores = {}
    for node in nodes:
        yi = Ei[nodes.index(node)]
        xi = Ni[nodes.index(node)]
        Cxi_theta = C * (xi ** theta)
        
        score = (max(yi, Cxi_theta) / min(yi, Cxi_theta)) * np.log(abs(yi - Cxi_theta) + 1)
        anomaly_scores[node] = score
    
    scores = np.array(list(anomaly_scores.values()))
    normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    lof = LocalOutlierFactor(n_neighbors=20)
    lof_scores = -lof.fit_predict(np.column_stack((Ei, Ni)))
    
    final_scores = normalized_scores + lof_scores
    
    final_anomaly_scores = {node: final_scores[i] for i, node in enumerate(nodes)}
    nx.set_node_attributes(G, final_anomaly_scores, 'anomaly_score')
    return final_anomaly_scores

def draw_graph(G, anomaly_scores):
    sorted_scores = sorted(anomaly_scores.items(), key=lambda x: x[1], reverse=True)
    top_10_nodes = [node for node, score in sorted_scores[:10]]
    
    node_colors = ['red' if node in top_10_nodes else 'blue' for node in G.nodes()]
    
    plt.figure(figsize=(12, 12))
    nx.draw(G, node_color=node_colors, with_labels=True)
    plt.show()

if __name__ == "__main__":
    file_path = '/Users/ptanasa/Desktop/lab6-ad/ca-AstroPh.txt'
    G = load_graph(file_path, max_lines=1500)
    features = extract_features(G)
    anomaly_scores = compute_anomaly_scores(G, features)
    draw_graph(G, anomaly_scores)
    print("Graph loaded with", G.number_of_nodes(), "nodes and", G.number_of_edges(), "edges.")