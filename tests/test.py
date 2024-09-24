from problems.image_deblurring import ImageDeblurring
from enums import Solver
import numpy as np
import math

class TestImageDeblurring():
    def __init__(self, A, x, rho, solvers):
        self.instance = ImageDeblurring(A, x, rho)
        self.solvers = solvers
        self.rel_tol = 10**-5
        self.abs_tol = 10**-10

    def solve_all(self, verbose=False):
        self.instance.solve_with_cvxpy(verbose)
        for solver in self.solvers:
            self.instance.canonicalize_and_solve(solver, verbose)

    def test_solutions(self):
        cvxpy_solution = self.instance.get_cvxpy_solution()
        for solver in self.solvers:
            solution = self.instance.get_solution(solver)
            assert math.isclose(cvxpy_solution[0], solution[0], rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            for i in range(len(solution[1])):
                assert math.isclose(cvxpy_solution[1][i], solution[1][i], rel_tol=self.rel_tol, abs_tol=self.abs_tol)

if __name__ == "__main__":
    A = np.array([[2, 0], [0, -1]])
    x = np.array([1, 0])
    rho = 3
    test = TestImageDeblurring(A, x, rho, [Solver.clarabel])
    test.solve_all(verbose=True)
    test.test_solutions()