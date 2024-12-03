from solution import Solution
from problems import problem
from enums import Solver
from scipy import sparse
import cvxpy as cp
import clarabel

class NetworkFlow(problem.Instance):
    def __init__(self, A, b, c, u):
        self.A = A
        self.c = c
        self.u = u

        self.n = len(self.c)

        self.P = None
        self.q = None
        self.D = None
        self.b = b
        self.cones = None
        self.constant_objective = None

        self.original_cvxpy_solution = None
        self.solutions = {}

    def solve_original_in_cvxpy(self, verbose=False):
        x = cp.Variable(self.n)
        objective = self.c.T @ x
        constraints = [self.A @ x == self.b, x >= 0, x <= self.u]
        problem = cp.Problem(cp.Minimize(objective), constraints)

        optimal_value = problem.solve(verbose=verbose)
        optimal_solution = x.value
        dual_solution = [constraint.dual_value for constraint in constraints]
        solve_time = problem.solver_stats.solve_time
        status = problem.status
        self.original_cvxpy_solution = Solution(Solver.cvxpy, 0, optimal_value,
                                                optimal_solution, None, dual_solution,
                                                solve_time, status)
        return self.original_cvxpy_solution
    
    def canonicalize(self):
        self.P = 0

        self.q = self.c

        I_n = sparse.identity(self.n)
        self.D = sparse.vstack([self.A, -I_n, I_n]).tocsc()

        self.cones = [clarabel.ZeroConeT(self.n), clarabel.NonnegativeConeT(2 * self.n)]

        self.constant_objective = 0