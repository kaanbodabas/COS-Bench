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

class Problem(Enum):
    NETWORK_FLOW = "Network Flow"
    IMAGE_DEBLURRING = "Image Deblurring"
    MAXCUT = "Maxcut"