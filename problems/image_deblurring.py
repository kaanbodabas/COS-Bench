from enums import Solver
from scipy import sparse
import numpy as np
import cvxpy as cp
import clarabel

class ImageDeblurring:
    def __init__(self, A, x, rho):
        self.A = A
        self.x = x
        self.rho = rho

        self.cvxpy_optimal_value = None
        self.cvxpy_original_image = None

        self.optimal_values = {}
        self.original_images = {}

    def solve_with_cvxpy(self, verbose=False):
        y = cp.Variable(len(self.x))
        objective = cp.norm(self.A @ y - self.x, 2)**2 + self.rho * cp.norm(y, 1)
        constraints = [y >= 0, y <= 1]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        self.cvxpy_optimal_value = problem.solve(verbose=verbose)
        self.cvxpy_original_image = y.value
        return self.cvxpy_optimal_value, self.cvxpy_original_image
    
    def get_cvxpy_solution(self):
        return self.cvxpy_optimal_value, self.cvxpy_original_image

    def canonicalize_and_solve(self, solver, verbose=False):
        if solver == Solver.clarabel:
            # canonicalize
            P = sparse.csc_matrix(2 * (self.A.T @ self.A))
            P = sparse.triu(P).tocsc()
            
            n = len(self.x)
            q = -2 * (self.A.T @ self.x) + self.rho * np.ones(n)
            
            I_n = sparse.identity(n)
            D = sparse.vstack([I_n, -I_n]).tocsc()
            
            b = np.concatenate([np.ones(n), np.zeros(n)])
            
            s = [clarabel.NonnegativeConeT(2 * n)]

            # solve
            settings = clarabel.DefaultSettings()
            settings.verbose = verbose
            problem = clarabel.DefaultSolver(P, q, D, b, s, settings)
            solution = problem.solve()
            optimal_value = solution.obj_val + self.x.T @ self.x
            original_image = solution.x
        
        self.optimal_values[solver] = optimal_value
        self.original_images[solver] = original_image
        return optimal_value, original_image

    def get_solution(self, solver):
        if solver not in self.optimal_values or solver not in self.original_images:
            raise Exception(f"Problem not yet solved with solver {solver}!")
        return self.optimal_values[solver], self.original_images[solver]