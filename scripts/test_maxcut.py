from utils import data, run, maps
from enums import Problem

solvers = maps.get_solvers("SDP")
csv_filename = "maxcut"
laplacian_matrices = [data.get_random_weighted_graph(3, 0.5)]
num_instances = 1
plot_title = "Maxcut Solve Times"

run.start(solvers, csv_filename, Problem.MAXCUT, (laplacian_matrices,))
run.results(csv_filename, solvers, num_instances, plot_title)