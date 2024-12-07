from solution import Solution
from problems import problem
from enums import Solver
from scipy import sparse
import numpy as np
import cvxpy as cp
import clarabel

class RobustPortfolio(problem.Instance):
    def __init__(self, l, u, mu, alpha, eta):
        self.l = l
        self.u = u
        self.mu = mu
        self.alpha = alpha
        self.eta = eta

        self.n = len(self.l)
        self.m = None

        self.P = None
        self.q = None
        self.D = None
        self.b = None
        self.cones = None
        self.constant_objective = None

        self.original_cvxpy_solution = None
        self.solutions = {}

    def solve_original_in_cvxpy(self, verbose=False):
        y = cp.Variable(self.n)
        objective = -self.mu @ y
        constraints = [self.mu @ y - np.sqrt(0.5 * np.log(1 / self.eta)) * cp.norm(np.diag(self.u - self.l) @ y, 2) >= self.alpha,
                       cp.sum(y) == 1, y >= 0]
        problem = cp.Problem(cp.Minimize(objective), constraints)

        optimal_value = problem.solve(verbose=verbose)
        optimal_solution = y.value
        dual_solution = [constraint.dual_value for constraint in constraints]
        solve_time = problem.solver_stats.solve_time
        status = problem.status
        self.original_cvxpy_solution = Solution(Solver.CVXPY, 0, optimal_value,
                                                optimal_solution, None, dual_solution,
                                                solve_time, status)
        return self.original_cvxpy_solution

    def canonicalize(self):
        self.P = sparse.csc_matrix((self.n, self.n))
        
        self.q = -self.mu
        
        self.D = -sparse.vstack([
            [np.ones(self.n)],
            np.identity(self.n),
            [self.mu],
            np.sqrt(0.5 * np.log(1 / self.eta)) * np.diag(self.u - self.l)], format="csc")
    
        self.b = np.hstack([-1, np.zeros(self.n), -self.alpha, np.zeros(self.n)])
        
        self.cones = [(1, clarabel.ZeroConeT(1)),
                      (self.n, clarabel.NonnegativeConeT(self.n)),
                      (self.n + 1, clarabel.SecondOrderConeT(self.n + 1))]
        self.m = 2 * self.n + 2

        self.constant_objective = 0