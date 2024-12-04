from utils import data, run, maps
from enums import Problem
import numpy as np

solvers = maps.get_solvers("LP")
csv_filename = "network_flow"
incidence_matrix, supply, cost, capacity = data.get_random_network(40, 0.3)
incidence_matrices = [incidence_matrix]
supplies = [supply]
costs = [cost]
capacities = [capacity]
num_instances = 1
plot_title = "Network Flow Solve Times"

run.start(solvers, csv_filename, Problem.NETWORK_FLOW, (incidence_matrices, supplies, costs, capacities))
run.results(csv_filename, solvers, num_instances, plot_title)