from scipy import sparse
import numpy as np
import constants
import osqp

SOLVED_STATUS = ["solved", "solved inaccurate"]

def solve(n, m, P, q, D, b, cones, verbose):
    stacked_P = sparse.block_diag([P, np.zeros((m, m))], format="csc")
    stacked_q = np.hstack([q, np.zeros(m)])
    stacked_D = sparse.vstack([
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
    problem = osqp.OSQP()
    problem.setup(stacked_P, stacked_q, stacked_D, lb, ub,
                  verbose=verbose, time_limit=constants.TIME_LIMIT)
    solution = problem.solve()

    status = solution.info.status
    if status in SOLVED_STATUS:
        optimal_value = solution.info.obj_val
        optimal_solution = solution.x[:n]
        primal_slacks = solution.x[n:]
        dual_solution = solution.y[:m]
        solve_time = solution.info.run_time
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, status)
    return (None, None, None, None, constants.TIME_LIMIT, status)
