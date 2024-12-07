from abc import ABC, abstractmethod
from solution import Solution
from utils import maps

class Instance(ABC):
    @abstractmethod
    def solve_original_in_cvxpy(self, verbose=False):
        pass

    def get_original_cvxpy_solution(self):
        if self.original_cvxpy_solution is None:
            raise Exception(f"Problem not yet solved with CVXPY!")
        return self.original_cvxpy_solution
    
    @abstractmethod
    def canonicalize(self):
        pass

    def solve(self, solver, verbose=False):
        if self.P is None:
            raise Exception(f"Problem not yet canonicalized!")
        
        solver_class = maps.get_solver_map()[solver]
        problem = (self.n, self.m, self.P, self.q, self.D, self.b, self.cones)
        solution_tuple = solver_class.solve(*problem, verbose)
        self.solutions[solver] = Solution(solver, self.constant_objective, *solution_tuple)
        return self.solutions[solver]
    
    def get_solution(self, solver):
        if solver not in self.solutions:
            raise Exception(f"Problem not yet solved with solver {solver}!")
        return self.solutions[solver]