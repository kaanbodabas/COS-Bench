from problems.robust_portfolio import RobustPortfolio
from utils import run
import numpy as np
import maps

solvers = maps.get_solvers("SOCP")
csv_filename = "robust_portfolio"
num_instances = 100
u = 0.05 + 0.6 * np.linspace(1, 1 / 500, 500)
mu = 1.05 + 0.3 * np.linspace(1, 1 / 500, 500)
us = [u] * num_instances
ls = [-u] * num_instances
mus = [mu] * num_instances
alphas = [1.1] * num_instances
etas = np.logspace(-4, 0, num_instances)
plot_title = "Robust Portfolio"

run.start(solvers, csv_filename, RobustPortfolio, (ls, us, mus, alphas, etas,))
run.results(csv_filename, solvers, num_instances, plot_title)