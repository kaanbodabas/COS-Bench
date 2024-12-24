from scipy import sparse
import coptpy as copt
import numpy as np
import constants

SOLVED_STATUS = [copt.COPT.OPTIMAL, copt.COPT.IMPRECISE]

def solve(n, m, P, q, D, b, cones, verbose):
    config = copt.EnvrConfig()
    if not verbose:
        config.set("nobanner", "1")
    env = copt.Envr(config)
    model = env.createModel()
    model.setParam(copt.COPT.Param.Logging, verbose)
    model.setParam(copt.COPT.Param.TimeLimit, constants.TIME_LIMIT)
    y = model.addMVar(shape=n, lb=-copt.COPT.INFINITY, ub=copt.COPT.INFINITY)
    s = model.addMVar(shape=m, lb=-copt.COPT.INFINITY, ub=copt.COPT.INFINITY)
    model.setObjective(0.5 * y @ P @ y + q @ y, sense=copt.COPT.MINIMIZE)
    model.addConstr(D @ y + s == b)
    i = 0
    for (dim, cone) in cones:
        cone = str(cone)
        if constants.ZERO_CONE in cone:
            model.addConstr(s[i:i + dim] == 0)
        elif constants.NONNEGATIVE_CONE in cone:
            model.addConstr(s[i:i + dim] >= 0)
        elif constants.SECOND_ORDER_CONE in cone:
            model.addCone(s[i:i + dim], copt.COPT.CONE_QUAD)
        elif constants.PSD_TRIANGLE_CONE in cone:
            S = model.addPsdVar(dim)
            rows, cols = np.tril_indices(dim)
            l = i
            for j, k in zip(rows, cols):
                indicator = np.zeros((dim, dim))
                indicator[j][k] = 1
                indicator[k][j] = 1
                indicator = model.addSparseMat(dim, *sparse.find(indicator))
                if j != k:
                    model.addConstr(np.sqrt(2) * indicator * S / 2 == s[l].item())
                else:
                    model.addConstr(indicator * S == s[l].item())
                l += 1
            dim = int(dim * (dim + 1) / 2)
        i += dim
    model.solve()

    status = model.status
    if status in SOLVED_STATUS:
        optimal_value = model.objval
        primal_variables = model.getValues()
        optimal_solution = primal_variables[:n]
        primal_slacks = primal_variables[n:]
        dual_solution = -np.array(model.getDuals())[:m]
        solve_time = model.getAttr(copt.COPT.Attr.SolvingTime)
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, status)
    return (None, None, None, None, constants.TIME_LIMIT, status)