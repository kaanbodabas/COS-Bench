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
        self.q = None
        self.D = None
        self.b = None
        self.cones = None

        self.solutions = {}

    def solve_with_cvxpy(self, verbose=False):
        y = cp.Variable(len(self.x))
        objective = cp.norm(self.A @ y - self.x, 2)**2 + self.rho * cp.norm(y, 1)
        constraints = [y <= 1, y >= 0]
        problem = cp.Problem(cp.Minimize(objective), constraints)

        optimal_value = problem.solve(verbose=verbose)
        optimal_solution = y.value
        dual_solution = [constraint.dual_value for constraint in constraints]
        solve_time = problem.solver_stats.solve_time
        status = problem.status

        self.solutions[Solver.cvxpy] = Solution(Solver.cvxpy, optimal_value, 0, optimal_solution, None, dual_solution, solve_time, status)
        return self.solutions[Solver.cvxpy]
    
    def get_cvxpy_solution(self):
        if Solver.cvxpy not in self.solutions:
            raise Exception(f"Problem not yet solved with CVXPY!")
        
        return self.solutions[Solver.cvxpy]

    def canonicalize(self, solver):
        if solver == Solver.clarabel:
            self.P = sparse.csc_matrix(2 * (self.A.T @ self.A))
            self.P = sparse.triu(self.P).tocsc()
            
            n = len(self.x)
            self.q = -2 * (self.A.T @ self.x) + self.rho * np.ones(n)
            
            I_n = sparse.identity(n)
            self.D = sparse.vstack([I_n, -I_n]).tocsc()
            
            self.b = np.concatenate([np.ones(n), np.zeros(n)])
            
            self.cones = [clarabel.NonnegativeConeT(2 * n)]

    def solve(self, solver, verbose=False):
        if self.P is None:
            raise Exception(f"Problem not yet canonicalized with solver {solver}!")
        
        if solver == Solver.cvxpy:
            raise Exception(f"Use 'solve_with_cvxpy' instead!")
        
        if solver == Solver.clarabel:
            settings = clarabel.DefaultSettings()
            settings.verbose = verbose
            # settings.equilibrate_enable = False
            problem = clarabel.DefaultSolver(self.P, self.q, self.D, self.b, self.cones, settings)

            solution = problem.solve()
            optimal_value = solution.obj_val
            constant_objective = self.x.T @ self.x
            optimal_solution = solution.x
            primal_slacks = solution.s
            dual_solution = solution.z
            solve_time = solution.solve_time
            status = solution.status
        
        self.solutions[solver] = Solution(solver, optimal_value, constant_objective, optimal_solution, primal_slacks, dual_solution, solve_time, status)
        return self.solutions[solver]

    def get_solution(self, solver):
        if solver not in self.solutions:
            raise Exception(f"Problem not yet solved with solver {solver}!")
        
        return self.solutions[solver]