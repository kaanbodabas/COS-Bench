from problems.robust_portfolio import RobustPortfolio
from utils import data, run, maps
import numpy as np

solvers = maps.get_solvers("SOCP")
csv_filename = "robust_portfolio"
l, u, mu = data.get_random_returns(200)
ls = np.array([l])
us = np.array([u])
mus = np.array([mu])
alphas = [1.1]
etas = [0.9]
num_instances = 1
plot_title = "Robust Portfolio Solve Times"

run.start(solvers, csv_filename, RobustPortfolio, (ls, us, mus, alphas, etas,))
run.results(csv_filename, solvers, num_instances, plot_title)