from enums import Solver, get_solve_map
from solution import Solution
from scipy import sparse
import numpy as np
import cvxpy as cp
import clarabel

# TODO: Standardize status reporting

class ImageDeblurring:
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

        self.original_cvxpy_solution = None
        self.solutions = {}

    def solve_original_in_cvxpy(self, verbose=False):
        y = cp.Variable(len(self.x))
        objective = cp.norm(self.A @ y - self.x, 2)**2 + self.rho * cp.norm(y, 1)
        constraints = [y <= 1, y >= 0]
        problem = cp.Problem(cp.Minimize(objective), constraints)

        optimal_value = problem.solve(verbose=verbose)
        optimal_solution = y.value
        dual_solution = [constraint.dual_value for constraint in constraints]
        solve_time = problem.solver_stats.solve_time
        status = problem.status

        self.original_cvxpy_solution = Solution(Solver.cvxpy, 0, optimal_value,
                                                optimal_solution, None, dual_solution,
                                                solve_time, status)
        return self.original_cvxpy_solution
    
    def get_original_cvxpy_solution(self):
        if self.original_cvxpy_solution is None:
            raise Exception(f"Problem not yet solved with CVXPY!")
        
        return self.original_cvxpy_solution 

    def canonicalize(self):
        self.P = sparse.csc_matrix(2 * (self.A.T @ self.A))
        
        self.q = -2 * (self.A.T @ self.x) + self.rho * np.ones(self.n)
        
        I_n = sparse.identity(self.n)
        self.D = sparse.vstack([I_n, -I_n]).tocsc()
        
        self.b = np.concatenate([np.ones(self.n), np.zeros(self.n)])
        
        self.cones = [clarabel.NonnegativeConeT(2 * self.n)]

    def solve(self, solver, verbose=False):
        if self.P is None:
            raise Exception(f"Problem not yet canonicalized!")
        
        constant_objective = self.x.T @ self.x

        solve = get_solve_map()[solver]
        problem = (self.n, self.P, self.q, self.D, self.b, self.cones)
        solution_tuple = solve(*problem, verbose)

        solution = Solution(solver, constant_objective, *solution_tuple)
        self.solutions[solver] = solution
        return solution

    def get_solution(self, solver):
        if solver not in self.solutions:
            raise Exception(f"Problem not yet solved with solver {solver}!")
        
        return self.solutions[solver]