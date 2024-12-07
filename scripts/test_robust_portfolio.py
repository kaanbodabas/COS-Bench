from problems.robust_portfolio import RobustPortfolio
from utils import data, run, maps
import numpy as np

solvers = maps.get_solvers("SOCP")
csv_filename = "robust_portfolio"
n = 200
mu = 1.05 + 0.3 * np.linspace(1, 1 / n, n)
u_max = 0.05 + 0.6 * np.linspace(1, 1 / n, n)  # Uncertainty sets
u_max[-1] = 0  # Cash no variability
epsilon = 1e-3  # Risk parameter (probability of failure)
alpha = 1.1 
ls = np.array([-u_max])
us = np.array([u_max])
mus = np.array([mu])
alphas = [alpha]
etas = [epsilon]
num_instances = 1
plot_title = "Robust Portfolio Solve Times"

run.start(solvers, csv_filename, RobustPortfolio, (ls, us, mus, alphas, etas,))
run.results(csv_filename, solvers, num_instances, plot_title)