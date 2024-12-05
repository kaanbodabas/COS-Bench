from solution import Solution
from problems import problem
from enums import Solver
from scipy import sparse
import cvxpy as cp
import numpy as np
import clarabel

class NetworkFlow(problem.Instance):
    def __init__(self, A, t, c, u):
        self.A = A
        self.t = t
        self.c = c
        self.u = u
        self.d = len(self.t)

        self.n = len(self.c)
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
        objective = self.c.T @ y
        constraints = [self.A @ y == self.t, y >= 0, y <= self.u]
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

        self.q = self.c

        I_n = sparse.identity(self.n)
        self.D = sparse.vstack([self.A, -I_n, I_n]).tocsc()

        self.b = np.hstack([self.t, np.zeros(self.n), self.u])

        self.cones = [(self.d, clarabel.ZeroConeT(self.d)),
                      (2 * self.n, clarabel.NonnegativeConeT(2 * self.n))]
        self.m = self.d + 2 * self.n

        self.constant_objective = 0