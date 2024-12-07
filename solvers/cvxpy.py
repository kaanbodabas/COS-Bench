from scipy import sparse
import numpy as np
import cvxpy as cp
import constants

SOLVED_STATUS = ["optimal", "optimal_inaccurate"]

# column-major order version of cp.vec_to_upper_tri
def vec_to_upper_tri(s, n, d):
    rows, cols = np.tril_indices(d)
    A_rows = rows + d * cols
    A_cols = np.arange(n)
    A_vals = np.ones(A_cols.size)
    A = sparse.csc_matrix((A_vals, (A_rows, A_cols)), shape=(d * d, n))
    return cp.reshape(A @ s, (d, d), order='F').T

def solve(n, m, P, q, D, b, cones, verbose):
    y = cp.Variable(n) 
    s = cp.Variable(m)
    objective = 0.5 * cp.quad_form(y, cp.psd_wrap(P)) + q @ y
    constraints = [D @ y + s == b]
    i = 0
    for (dim, cone) in cones:
        cone = str(cone)
        if constants.ZERO_CONE in cone:
            constraints.append(s[i:i + dim] == 0)
        elif constants.NONNEGATIVE_CONE in cone:
            constraints.append(s[i:i + dim] >= 0)
        elif constants.SECOND_ORDER_CONE in cone:
            constraints.append(cp.norm(s[i + 1:i + dim], 2) <= s[i])
        elif constants.PSD_TRIANGLE_CONE in cone:
            vec_dim = int(dim * (dim + 1) / 2)
            S = vec_to_upper_tri(s[i:i + vec_dim], n, dim)
            diag = cp.diag(cp.diag(S))
            S = (S - diag) / np.sqrt(2) + diag
            S = S + S.T - diag
            constraints.append(S >> 0)
            dim = vec_dim
        i += dim
    problem = cp.Problem(cp.Minimize(objective), constraints)

    optimal_value = problem.solve(verbose=verbose)
    status = problem.status
    if status in SOLVED_STATUS:
        optimal_solution = y.value
        primal_slacks = s.value
        dual_solution = constraints[0].dual_value
        solve_time = problem.solver_stats.solve_time
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, status)
    return (None, None, None, None, constants.TIME_LIMIT, status)