from enum import Enum

class Solver(Enum):
    cvxpy = "CVXPY"
    clarabel = "CLARABEL"
    gurobi = "GUROBI"
    mosek = "MOSEK"
    # etc
