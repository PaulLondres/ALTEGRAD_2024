"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs, inv
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk


# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

nx.draw_networkx(G)

############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[node]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy with DeepWalk embeddings : ", accuracy_score(y_test, y_pred))

############## Task 8
#Get the adjacency matrix of the graph G
def get_spectral_embeddings(graph, dim_eig_vec):
    W = nx.adjacency_matrix(graph).astype(float)

    #Compute the degree matrix
    degrees = np.array(W.sum(axis=1)).flatten()
    D = diags(degrees)
    #print(D)
    #Compute the unnormalized Laplacian matrix L = Id - D^-1*W
    L = eye(len(degrees)) - inv(D).dot(W)

    #Calculate the first k eigenvectors of the Laplacian matrix
    _, eigvecs = eigs(L, k=dim_eig_vec, which='SR')
    eigvecs = eigvecs.real
    return eigvecs
spectral_embeddings = get_spectral_embeddings(G, 2)
X_train_spec = spectral_embeddings[idx_train,:]
X_test_spec = spectral_embeddings[idx_test,:]

clf_spec = LogisticRegression(solver='lbfgs')
clf_spec.fit(X_train_spec, y_train)
y_pred_spec = clf_spec.predict(X_test_spec)

print("Accuracy with Spectral embeddings : ", accuracy_score(y_test, y_pred_spec))

