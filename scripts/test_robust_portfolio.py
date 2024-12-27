from problems.robust_portfolio import RobustPortfolio
from utils import run
import numpy as np
import maps

solvers = maps.get_solvers("SOCP")
csv_filename = "robust_portfolio"
num_instances = 10000
us = []
ls = []
mus = []
for _ in range(num_instances):
    u = np.random.uniform(0, 0.6, 200)
    mu = np.random.normal(1.05, 0.2, 200)
    us.append(u)
    ls.append(-u)
    mus.append(mu)
alphas = [1.1] * num_instances
etas = np.logspace(-4, 0, num_instances)
plot_title = "Robust Portfolio"

run.start(solvers, csv_filename, RobustPortfolio, (ls, us, mus, alphas, etas,))
run.results(csv_filename, solvers, num_instances, plot_title)