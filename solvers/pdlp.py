from ortools.pdlp import solvers_pb2, solve_log_pb2
from ortools.pdlp.python import pdlp
from scipy import sparse
import numpy as np
import constants

code = solve_log_pb2.TerminationReason
SOLVED_STATUS = [code.TERMINATION_REASON_OPTIMAL]

def solve(n, m, P, q, D, b, cones, verbose):
    problem = pdlp.QuadraticProgram()
    problem.objective_vector = np.hstack([q, np.zeros(m)])
    problem.variable_lower_bounds = -np.inf * np.ones(n + m)
    problem.variable_upper_bounds = np.inf * np.ones(n + m)
    problem.constraint_matrix = sparse.vstack([
        sparse.hstack([D, sparse.identity(m)]),
        sparse.hstack([np.zeros((m, n)), sparse.identity(m)])], format="csc")
    cone_lb = []
    cone_ub = []
    for (dim, cone) in cones:
        cone = str(cone)
        if constants.ZERO_CONE in cone:
            cone_lb.extend(np.zeros(dim))
            cone_ub.extend(np.zeros(dim))
        elif constants.NONNEGATIVE_CONE in cone:
            cone_lb.extend(np.zeros(dim))
            cone_ub.extend(np.inf * np.ones(dim))
    lb = np.hstack([b, cone_lb])
    ub = np.hstack([b, cone_ub])
    problem.constraint_lower_bounds = lb
    problem.constraint_upper_bounds = ub
    settings = solvers_pb2.PrimalDualHybridGradientParams()
    settings.verbosity_level = verbose
    settings.termination_criteria.time_sec_limit = constants.TIME_LIMIT
    solution = pdlp.primal_dual_hybrid_gradient(problem, settings)
    
    status = solution.solve_log.termination_reason
    if status in SOLVED_STATUS:
        optimal_value = solution.solve_log.solution_stats.convergence_information[0].primal_objective
        optimal_solution = solution.primal_solution[:n]
        primal_slacks = solution.primal_solution[n:]
        dual_solution = -solution.dual_solution[:m]
        solve_time = solution.solve_log.solution_stats.cumulative_time_sec
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, status)
    return (None, None, None, None, constants.TIME_LIMIT, status)