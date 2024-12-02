from utils import solve
from enum import Enum

class Solver(Enum):
    CVXPY = "CVXPY"
    CLARABEL = "CLARABEL"
    GUROBI = "GUROBI"
    MOSEK = "MOSEK"
    OSQP = "OSQP"
    PDLP = "PDLP"
    SCS = "SCS"
    SDPA = "SPDA"
    COPT = "COPT"

    def __str__(self):
        return self.value

def get_solvers(problem_class):
    if problem_class == "LP":
        return [Solver.CVXPY, Solver.CLARABEL, Solver.GUROBI, Solver.MOSEK, Solver.OSQP, Solver.PDLP, Solver.SCS, Solver.SDPA, Solver.COPT]
    elif problem_class == "QP":
        return [Solver.CVXPY, Solver.CLARABEL, Solver.GUROBI, Solver.MOSEK, Solver.OSQP, Solver.SCS, Solver.COPT]
    elif problem_class == "SOCP":
        return [Solver.CVXPY, Solver.CLARABEL, Solver.GUROBI, Solver.MOSEK, Solver.SCS, Solver.COPT]
    elif problem_class == "SDP":
        return [Solver.CVXPY, Solver.CLARABEL, Solver.MOSEK, Solver.SCS, Solver.SDPA, Solver.COPT]
    else:
        raise Exception("Problem class {problem_class} not supported!")

def get_solve_map():
    return {Solver.CVXPY: solve.with_cvxpy,
            Solver.CLARABEL: solve.with_clarabel,
            Solver.GUROBI: solve.with_gurobi,
            Solver.MOSEK: solve.with_mosek,
            Solver.OSQP: solve.with_osqp,
            Solver.PDLP: solve.with_pdlp,
            Solver.SCS: solve.with_scs}