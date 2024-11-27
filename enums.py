from utils import solve
from enum import Enum

class Solver(Enum):
    CVXPY = "CVXPY"
    CLARABEL = "CLARABEL"
    GUROBI = "GUROBI"
    MOSEK = "MOSEK"
    OSQP = "OSQP"
    SCS = "SCS"
    COPT = "COPT"
    CVXOPT = "CVXOPT"
    SDPA = "SPDA"
    PDLP = "PDLP"

    def __str__(self):
        return self.value

def get_qp_solvers():
    return [Solver.CVXPY, Solver.CLARABEL, Solver.GUROBI, Solver.MOSEK, Solver.OSQP]

def get_qp_solve_map():
    return {Solver.CVXPY: solve.with_cvxpy,
            Solver.CLARABEL: solve.with_clarabel,
            Solver.GUROBI: solve.with_gurobi,
            Solver.MOSEK: solve.with_mosek,
            Solver.OSQP: solve.with_osqp}