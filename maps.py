from solvers import cvxpy, clarabel, gurobi, mosek, osqp, pdlp, scs, copt
from enums import Solver

def get_solvers(problem_class):
    if problem_class == "LP":
        return [Solver.CLARABEL, Solver.GUROBI, Solver.MOSEK, Solver.OSQP, Solver.PDLP, Solver.SCS, Solver.COPT]
    elif problem_class == "QP":
        return [Solver.CLARABEL, Solver.GUROBI, Solver.MOSEK, Solver.OSQP, Solver.SCS, Solver.COPT]
    elif problem_class == "SOCP":
        return [Solver.CLARABEL, Solver.GUROBI, Solver.MOSEK, Solver.SCS, Solver.COPT]
    elif problem_class == "SDP":
        return [Solver.CLARABEL, Solver.MOSEK, Solver.SCS, Solver.COPT]
    else:
        raise Exception("Problem class {problem_class} not supported!")

def get_solver_map():
    return {Solver.CVXPY: cvxpy,
            Solver.CLARABEL: clarabel,
            Solver.GUROBI: gurobi,
            Solver.MOSEK: mosek,
            Solver.OSQP: osqp,
            Solver.PDLP: pdlp,
            Solver.SCS: scs,
            Solver.COPT: copt}