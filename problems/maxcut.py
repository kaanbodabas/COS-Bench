from solution import Solution
from problems import problem
from enums import Solver
from scipy import sparse
import cvxpy as cp
import numpy as np
import clarabel

def vec_for_inner_product(C):
    d = len(C)
    c = C * 2
    for i in range(d):
        c[i][i] /= 2
    return c[np.tril_indices(d)]
    
def triu_identity(d, n):
    A = np.zeros((d, n))
    j = 0
    for i in range(d):
        A[i][j] = 1
        j += i + 2
    return A

def to_vec(n):
    B = np.zeros((n, n))
    j = 0
    diag = [0]
    for i in range(n):
        if i in diag:
            B[i][i] = 1
        else:
            B[i][i] = np.sqrt(2)
        j += i + 2
        diag.append(j)
    return B

class Maxcut(problem.Instance):
    def __init__(self, C):
        self.C = C
        self.d = len(C)

        self.n = int(self.d * (self.d + 1) / 2)
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
        Y = cp.Variable((self.d, self.d))
        objective = -cp.trace(self.C @ Y)
        constraints = [Y >> 0]
        for i in range(self.d):
            constraints.append(Y[i][i] == 1)
        problem = cp.Problem(cp.Minimize(objective), constraints)

        optimal_value = problem.solve(verbose=verbose)
        optimal_solution = Y.value
        dual_solution = [constraint.dual_value for constraint in constraints]
        solve_time = problem.solver_stats.solve_time
        status = problem.status
        self.original_cvxpy_solution = Solution(Solver.CVXPY, 0, optimal_value,
                                                optimal_solution, None, dual_solution,
                                                solve_time, status)
        return self.original_cvxpy_solution
    
    def canonicalize(self):
        self.P = sparse.csc_matrix((self.n, self.n))

        self.q = -vec_for_inner_product(self.C)

        self.D = sparse.vstack([triu_identity(self.d, self.n), -to_vec(self.n)]).tocsc()

        self.b = np.hstack([np.ones(self.d), np.zeros(self.n)])

        self.cones = [(self.d, clarabel.ZeroConeT(self.d)),
                      (self.d, clarabel.PSDTriangleConeT(self.d))]
        self.m = self.d + self.n

        self.constant_objective = 0