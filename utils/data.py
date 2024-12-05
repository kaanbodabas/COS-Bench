from scipy import linalg as la
import networkx as nx
import numpy as np
import emnist

def get_emnist_training_images(amt, subset="letters"):
    images, _ = emnist.extract_training_samples(subset)
    column_stacked_images = [image.flatten(order="F") for image in images]
    return np.array(column_stacked_images[:amt]) / 255

# https://github.com/stellatogrp/data_driven_optimizer_guarantees/blob/main/opt_guarantees/examples/mnist.py
def get_2D_blur_matrix(m, n, width):

    def get_blur_matrix(m, width):
        half_length = int(np.ceil((width-1)/2))    
        rows, cols = np.zeros(m), np.zeros(m)
        cols[:1 + half_length] = 1 / width
        rows[:1 + half_length] = 1 / width
        return la.toeplitz(cols, rows)

    blur_cols = get_blur_matrix(m, width)
    blur_rows = get_blur_matrix(n, width)
    return np.kron(blur_rows, blur_cols)

def get_rho_range(n, lb, ub):
    pass

def get_random_network(n, p):
    G = nx.DiGraph()
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    for u in range(n):
        for v in range(n):
            if u != v and np.random.random() < p:
                G.add_edge(u, v)
    incidence_matrix = np.array(nx.incidence_matrix(G, oriented=True).toarray())
    
    supply = []
    for node in range(n):
        net_degree = G.out_degree(node) - G.in_degree(node)
        if net_degree > 0:
            supply.append(np.random.randint(0, 25))
        elif net_degree < 0:
            supply.append(-np.random.randint(0, 25))
        else:
            supply.append(0)
    extra = sum(supply)
    for i in range(n):
        while supply[i] > 0 and extra > 0:
            supply[i] -= 1
            extra -= 1
        while supply[i] < 0 and extra < 0:
            supply[i] += 1
            extra += 1
        if extra == 0:
            break

    cost = []
    capacity = []
    for _, _ in G.edges():
        cost.append(np.random.randint(0, 10))
        capacity.append(np.random.randint(10, 30))

    return incidence_matrix, np.array(supply), np.array(cost), np.array(capacity)

def get_random_weighted_graph(n, p):
    random_graph = nx.erdos_renyi_graph(n, p)
    for (u, v) in random_graph.edges():
        random_graph[u][v]["weight"] = np.random.randint(0, 25)
    return np.array(nx.laplacian_matrix(random_graph).toarray())