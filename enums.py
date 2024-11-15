from enum import Enum

class Solver(Enum):
    CVXPY = "CVXPY"
    CLARABEL = "CLARABEL"
    GUROBI = "GUROBI"
    MOSEK = "MOSEK"
    OSQP = "OSQP"

    def __str__(self):
        return self.value

def get_qp_solvers():
    return [Solver.CVXPY, Solver.CLARABEL, Solver.GUROBI]