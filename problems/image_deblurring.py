from solution import Solution
from problems import problem
from enums import Solver
from scipy import sparse
import numpy as np
import cvxpy as cp
import clarabel

class ImageDeblurring(problem.Instance):
    def __init__(self, A, x, rho):
        self.A = A
        self.x = x
        self.rho = rho

        self.n = len(x)

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
        objective = cp.norm(self.A @ y - self.x, 2)**2 + self.rho * cp.norm(y, 1)
        constraints = [y <= 1, y >= 0]
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
        self.P = sparse.csc_matrix(2 * (self.A.T @ self.A))
        
        self.q = -2 * (self.A.T @ self.x) + self.rho * np.ones(self.n)
        
        I_n = sparse.identity(self.n)
        self.D = sparse.vstack([I_n, -I_n]).tocsc()
        
        self.b = np.hstack([np.ones(self.n), np.zeros(self.n)])
        
        self.cones = [clarabel.NonnegativeConeT(2 * self.n)]

        self.constant_objective = self.x.T @ self.x