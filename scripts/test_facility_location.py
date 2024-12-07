from utils import data, run, maps
from enums import Problem
import numpy as np

solvers = maps.get_solvers("SOCP")
csv_filename = "facility_location"
v_arrays = [np.array([[0, 1, 2, 3], [2, 4, 5, 6], [0, 1, 2, 8]])]
num_instances = 1
plot_title = "Facility Location Solve Times"

run.start(solvers, csv_filename, Problem.FACILITY_LOCATION, (v_arrays,))
# run.results(csv_filename, solvers, num_instances, plot_title)