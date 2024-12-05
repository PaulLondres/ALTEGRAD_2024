"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1
network = nx.read_edgelist("../datasets/CA-HepTh.txt", delimiter='\t', comments='#')
print("Number of nodes : ", network.number_of_nodes(), "Number of edges : ", network.number_of_edges())
############## Task 2
print("Number of connected components : ", nx.number_connected_components(network))
gcc_nodes = max(nx.connected_components(network))
gcc = network.subgraph(gcc_nodes)
print("Number of nodes in giant component : ", gcc.number_of_nodes(), "Number of edges in giant component : ", gcc.number_of_edges())
print("Fraction of nodes in giant components : ", gcc.number_of_nodes()/network.number_of_nodes())
print("Fraction of edges in giant components : ", gcc.number_of_edges()/network.number_of_edges())



