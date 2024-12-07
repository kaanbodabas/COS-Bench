from problems.maxcut import Maxcut
from utils import data, run, maps

solvers = maps.get_solvers("SDP")
csv_filename = "maxcut"
laplacian_matrices = [data.get_random_weighted_graph(40, 0.5, 0, 25)]
num_instances = 1
plot_title = "Maxcut Solve Times"

run.start(solvers, csv_filename, Maxcut, (laplacian_matrices,))
run.results(csv_filename, solvers, num_instances, plot_title)