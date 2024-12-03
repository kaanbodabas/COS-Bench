from utils import data, run, maps
from enums import Problem
import numpy as np

solvers = maps.get_solvers("LP")
csv_filename = "network_flow"
# incidence_matrix, cost, capacity = data.get_random_network(10, 1)
# incidence_matrices = [incidence_matrix]
# costs = [cost]
# capacities = [capacity]
# supplies = [data.get_random_supply(-10, 10, len(incidence_matrix))]
incidence_matrices = [np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                       [0, 0, 1, 1, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 1],
                       [-1, 0, -1, 0, 0, 0, 0, 0],
                       [0, -1, 0, -1, 0, -1, -1, 0],
                       [0, 0, 0, 0, -1, 0, 0, -1]])]
costs = [np.array([5, 6, 8, 4, 3, 9, 3, 6])]
capacities = [np.array([20, 20, 20, 20, 20, 20, 20, 20])]
supplies = [np.array([7, 11, 18, 12, -10, -23, -15])]
num_instances = 1
plot_title = "Network Flow Solve Times"

run.start(solvers, csv_filename, Problem.NETWORK_FLOW, (incidence_matrices, supplies, costs, capacities))
run.results(csv_filename, solvers, num_instances, plot_title)