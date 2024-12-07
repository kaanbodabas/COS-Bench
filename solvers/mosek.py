from utils import psd_helper
from enums import Solver
from scipy import sparse
import numpy as np
import constants
import mosek

SOLVED_STATUS = [mosek.solsta.optimal,
                mosek.solsta.prim_and_dual_feas,
                mosek.solsta.integer_optimal]

def solve(n, m, P, q, D, b, cones, verbose):
    env = mosek.Env()
    task = env.Task()
    task.putintparam(mosek.iparam.log, int(verbose))
    task.putdouparam(mosek.dparam.optimizer_max_time, constants.TIME_LIMIT)
    task.appendcons(m)
    task.appendvars(n + m)
    task.putclist(list(range(n)), q)
    task.putvarboundsliceconst(0, n, mosek.boundkey.fr, -np.inf, np.inf)
    P = sparse.tril(P, format="coo")
    task.putqobj(P.row, P.col, P.data)
    stacked_D = sparse.hstack([D, sparse.identity(m)])
    task.putaijlist(*sparse.find(stacked_D))
    task.putconboundslice(0, m, [mosek.boundkey.fx] * m, b, b)
    i = n
    j = 0
    soc_infos = []
    psd_infos = []
    for (dim, cone) in cones:
        cone = str(cone)
        if constants.ZERO_CONE in cone:
            task.putvarboundsliceconst(i, i + dim, mosek.boundkey.fx, 0, 0)
        elif constants.NONNEGATIVE_CONE in cone:
            task.putvarboundsliceconst(i, i + dim, mosek.boundkey.lo, 0, np.inf)
        elif constants.SECOND_ORDER_CONE in cone:
            # TODO
            soc_infos.append((i - n, j, dim))
            task.putvarboundsliceconst(i, i + dim, mosek.boundkey.fr, -np.inf, np.inf)
            task.appendafes(dim)
            F = sparse.identity(dim)
            rows, cols, vals = sparse.find(F)
            task.putafefentrylist(rows + j, cols + j, vals)
            domain = task.appendquadraticconedomain(dim)
            task.appendacc(domain, list(range(j, j + dim)), None)
            j += dim
        elif constants.PSD_TRIANGLE_CONE in cone:
            vec_dim = int(dim * (dim + 1) / 2)
            psd_infos.append((i - n, j, vec_dim, dim))
            task.putvarboundsliceconst(i, i + vec_dim, mosek.boundkey.fr, -np.inf, np.inf)
            task.appendafes(vec_dim)
            F = psd_helper.send_triu_vec_to_tril_vec(vec_dim, dim, Solver.MOSEK)
            rows, cols, vals = sparse.find(F)
            task.putafefentrylist(rows + j, cols + j, vals)
            domain = task.appendsvecpsdconedomain(vec_dim)
            task.appendacc(domain, list(range(j, j + vec_dim)), None)
            j += vec_dim
            dim = vec_dim
        i += dim
    task.putobjsense(mosek.objsense.minimize)
    task.optimize()

    status = task.getsolsta(mosek.soltype.itr)
    if status in SOLVED_STATUS:
        optimal_value = task.getprimalobj(mosek.soltype.itr)
        optimal_solution = task.getxxslice(mosek.soltype.itr, 0, n)
        primal_slacks = task.getxxslice(mosek.soltype.itr, n, n + m)
        dual_solution = -np.array(task.gety(mosek.soltype.itr))
        cone_dual_solution = np.array(task.getaccdotys(mosek.soltype.itr))
        for soc_info in soc_infos:
            i, j, dim = soc_info
            dual_solution[i:i + dim] += cone_dual_solution[j:j + dim]
        for psd_info in psd_infos:
            i, j, vec_dim, dim = psd_info
            F = psd_helper.send_triu_vec_to_tril_vec(vec_dim, dim, None)
            dual_solution[i:i + vec_dim] = F.T @ (dual_solution[i:i + vec_dim] + cone_dual_solution[j:j + vec_dim])
        solve_time = task.getdouinf(mosek.dinfitem.optimizer_time)
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, status)
    return (None, None, None, None, constants.TIME_LIMIT, status)