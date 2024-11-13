from solution import Solution
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

        self.P = None
        self.P_triu = None
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

        self.original_cvxpy_solution = Solution(Solver.cvxpy, optimal_value, 0,
                                                optimal_solution, None, dual_solution,
                                                solve_time, status)
        return self.original_cvxpy_solution
    
    def get_original_cvxpy_solution(self):
        if self.original_cvxpy_solution is None:
            raise Exception(f"Problem not yet solved with CVXPY!")
        
        return self.original_cvxpy_solution 

    def canonicalize(self):
        self.P = sparse.csc_matrix(2 * (self.A.T @ self.A))
        self.P_triu = sparse.triu(self.P).tocsc()
        
        n = len(self.x)
        self.q = -2 * (self.A.T @ self.x) + self.rho * np.ones(n)
        
        I_n = sparse.identity(n)
        self.D = sparse.vstack([I_n, -I_n]).tocsc()
        
        self.b = np.concatenate([np.ones(n), np.zeros(n)])
        
        self.cones = [clarabel.NonnegativeConeT(2 * n)]

    def solve(self, solver, verbose=False):
        if self.P is None:
            raise Exception(f"Problem not yet canonicalized!")
        
        if solver == Solver.cvxpy:
            n = len(self.x)
            y = cp.Variable(n)
            s = cp.Variable(2 * n)
            objective = 0.5 * cp.quad_form(y, cp.psd_wrap(self.P)) + self.q @ y
            constraints = [self.D @ y + s == self.b, s >= 0]
            problem = cp.Problem(cp.Minimize(objective), constraints)

            optimal_value = problem.solve(verbose=verbose)
            constant_objective = self.x.T @ self.x
            optimal_solution = y.value
            primal_slacks = s.value
            dual_solution = constraints[0].dual_value
            solve_time = problem.solver_stats.solve_time
            status = problem.status
        
        if solver == Solver.clarabel:
            settings = clarabel.DefaultSettings()
            settings.verbose = verbose
            # settings.equilibrate_enable = False
            problem = clarabel.DefaultSolver(self.P_triu, self.q, self.D, self.b, self.cones, settings)
            solution = problem.solve()
            
            optimal_value = solution.obj_val
            constant_objective = self.x.T @ self.x
            optimal_solution = solution.x
            primal_slacks = solution.s
            dual_solution = solution.z
            solve_time = solution.solve_time
            status = solution.status
        
        solution = Solution(solver, optimal_value, constant_objective,
                            optimal_solution, primal_slacks, dual_solution,
                            solve_time, status)
        self.solutions[solver] = solution
        return solution

    def get_solution(self, solver):
        if solver not in self.solutions:
            raise Exception(f"Problem not yet solved with solver {solver}!")
        
        return self.solutions[solver]