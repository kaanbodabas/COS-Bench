from problems.image_deblurring import ImageDeblurring
from utils import data, verify
from enums import Solver
import numpy as np

class BenchmarkImageDeblurring():
    def __init__(self, A, x, rho, solvers):
        self.problem = ImageDeblurring(A, x, rho)
        self.solvers = solvers
        self.eps_primal = 10**-5
        self.eps_dual = 10**-5
        # self.eps_slackness = 10**-5
        self.eps_duality_gap = 10**-5

    def solve_all(self, verbose=False):
        self.problem.solve_with_cvxpy(verbose)
        for solver in self.solvers:
            self.problem.canonicalize(solver)
            self.problem.solve(solver, verbose)

    def verify_solutions(self):
        cvxpy_solution = self.problem.get_cvxpy_solution()
        clarabel_solution = self.problem.get_solution(Solver.clarabel)
        # print(cvxpy_solution, clarabel_solution)
        verify.is_qp_solution_optimal(self.problem, clarabel_solution, self.eps_primal, self.eps_dual, self.eps_duality_gap)

if __name__ == "__main__":
    A = data.get_2D_blur_matrix(28, 28, 8)
    x = data.get_emnist_images()
    rho = 1
    test = BenchmarkImageDeblurring(A, x, rho, [Solver.clarabel])
    test.solve_all(verbose=False)
    test.verify_solutions()