from problems.network_flow import NetworkFlow
from utils import data, run
import maps

solvers = maps.get_solvers("LP")
csv_filename = "network_flow"
num_instances = 2
incidence_matrices = []
supplies = []
costs = []
capacities = []
for _ in range(num_instances):
    incidence_matrix, supply, cost, capacity = data.get_random_network(30, 0.5, 25, 0, 10, 0, 30)
    incidence_matrices.append(incidence_matrix)
    supplies.append(supply)
    costs.append(cost)
    capacities.append(capacity)
plot_title = "Network Flow"

run.start(solvers, csv_filename, NetworkFlow, (incidence_matrices, supplies, costs, capacities))
run.results(csv_filename, solvers, num_instances, plot_title)