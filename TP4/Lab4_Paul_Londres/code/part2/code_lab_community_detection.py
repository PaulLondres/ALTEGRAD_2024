"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs, inv
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    #Get the adjacency matrix of the graph G
    W = nx.adjacency_matrix(G).astype(float)

    #Compute the degree matrix
    degrees = np.array(W.sum(axis=1)).flatten()
    D = diags(degrees)
    #print(D)
    #Compute the unnormalized Laplacian matrix L = Id - D^-1*W
    L = eye(len(degrees)) - inv(D).dot(W)

    #Calculate the first k eigenvectors of the Laplacian matrix
    _, eigvecs = eigs(L, k=k, which='SR')
    eigvecs = eigvecs.real

    #Apply k-means clustering on the rows of the eigenvector matrix
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(eigvecs)

    #Assign each node in G to a cluster
    clustering = {node: labels[i] for i, node in enumerate(G.nodes())}

    return clustering

############## Task 4
network = nx.read_edgelist("../datasets/CA-HepTh.txt", delimiter='\t', comments='#')
gcc_nodes = max(nx.connected_components(network))
gcc = network.subgraph(gcc_nodes)
clusters = spectral_clustering(gcc, 50)

############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):

    m = G.number_of_edges()

    clusters = set(clustering.values())
    #Initialize modularity
    modularity = 0

    for cluster in clusters:
      nodes_in_cluster = [node for node in G.nodes() if clustering[node]==cluster]
      cluster_subG = G.subgraph(nodes_in_cluster)
      lc = cluster_subG.number_of_edges()

      dc = sum([G.degree(node) for node in nodes_in_cluster])

      modularity+=lc/m - (dc/(2*m))**2

    return modularity



############## Task 6

print(modularity(gcc, clusters))
random_clusters = {}
for node in gcc.nodes():
  random_clusters[node] = randint(0, 49)
print(modularity(gcc, random_clusters))



