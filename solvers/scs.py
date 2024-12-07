from utils import psd_helper
from enums import Solver
from scipy import sparse
import numpy as np
import constants
import scs

SOLVED_STATUS = ["solved", "solved inaccurate"]

def solve(n, m, P, q, D, b, cones, verbose):
    stacked_P = sparse.block_diag([P, np.zeros((m, m))], format="csc")
    stacked_q = np.hstack([q, np.zeros(m)])
    stacked_D = sparse.vstack([
        sparse.hstack([D, sparse.identity(m)]), 
        sparse.hstack([np.zeros((m, n)), -sparse.identity(m)])], format="csc")
    z = 0
    l = 0
    q_array = []
    s = 0
    i = m
    j = n
    for (dim, cone) in cones:
        cone = str(cone)
        if constants.ZERO_CONE in cone:
            z += dim
        elif constants.NONNEGATIVE_CONE in cone:
            l += dim
        elif constants.SECOND_ORDER_CONE in cone:
            q_array.append(dim)
        elif constants.PSD_TRIANGLE_CONE in cone:
            vec_dim = int(dim * (dim + 1) / 2)
            stacked_D = stacked_D.tolil()
            stacked_D[i:i + vec_dim, j:j + vec_dim] = -psd_helper.send_triu_vec_to_tril_vec(vec_dim, dim, Solver.SCS)
            stacked_D = stacked_D.tocsc()
            s += dim
            dim = vec_dim
        i += dim
        j += dim
    ub = np.hstack([b, np.zeros(m)])
    data = dict(P=stacked_P, A=stacked_D, b=ub, c=stacked_q)
    cone = dict(z=m + z, l=l, q=q_array, s=s)
    problem = scs.SCS(data, cone, verbose=verbose, time_limit_secs=constants.TIME_LIMIT)
    solution = problem.solve()

    status = solution["info"]["status"]
    if status in SOLVED_STATUS:
        optimal_value = solution["info"]["pobj"]
        optimal_solution = solution["x"][:n]
        primal_slacks = solution["x"][n:]
        dual_solution = solution["y"][:m]
        solve_time = solution["info"]["solve_time"] / 1000
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, status)
    return (None, None, None, None, constants.TIME_LIMIT, status)