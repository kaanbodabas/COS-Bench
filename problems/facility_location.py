from solution import Solution
from problems import problem
from enums import Solver
from scipy import sparse
import numpy as np
import cvxpy as cp
import clarabel

class FacilityLocation(problem.Instance):
    def __init__(self, v_array):
        self.v_array = v_array
        self.f = len(self.v_array)
        self.d = len(self.v_array[0])

        self.n = self.f + self.d
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
        objective = cp.sum(y[:self.f])
        constraints = []
        for i, v in enumerate(self.v_array):
            constraints.append(cp.norm(v - y[self.f:], 2) <= y[i])
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
        
        self.q = np.hstack([np.ones(self.f), np.zeros(self.d)])
        
        I_f = np.identity(self.f)
        O_df = np.zeros((self.d, self.f))
        I_d = np.identity(self.d)
        self.D = None
        for i in range(self.f):
            if self.D is None:
                self.D = np.vstack([
                    np.hstack([-I_f[i], np.zeros(self.d)]),
                    np.hstack([O_df, I_d])])
            else:
                self.D = np.vstack([self.D,
                    np.hstack([-I_f[i], np.zeros(self.d)]),
                    np.hstack([O_df, I_d])])
        self.D = sparse.csc_matrix(self.D)
    
        self.b = []
        for i in range(self.f):
            self.b.append(0)
            self.b.extend(self.v_array[i])
        self.b = np.array(self.b)
        
        self.cones = []
        for i in range(self.f):
            self.cones.append((self.d + 1, clarabel.SecondOrderConeT(self.d + 1)))
        self.m = self.f * (self.d + 1)

        self.constant_objective = 0