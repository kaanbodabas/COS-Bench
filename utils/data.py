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

def get_random_network(nodes, edge_probability):
    network = nx.erdos_renyi_graph(nodes, edge_probability)
    costs = []
    capacities = []
    for u, v in network.edges():
        # numbers are arbitrary for now
        costs.append(5)
        capacities.append(100)
    incidence_matrix = np.array(nx.incidence_matrix(network, oriented=True).toarray())
    return incidence_matrix, np.array(costs), np.array(capacities)

def get_random_supply(lb, ub, n):
    random_vector = np.random.randint(lb, ub, n)
    random_vector[0] -= sum(random_vector)
    return random_vector