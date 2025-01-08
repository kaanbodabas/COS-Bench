import gurobipy as gp
import constants

SOLVED_STATUS = [2, 13]

def solve(n, m, P, q, D, b, cones, verbose):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", verbose)
    env.setParam("TimeLimit", constants.TIME_LIMIT)
    env.start()
    model = gp.Model(env=env)
    y = model.addMVar(shape=n, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY) 
    s = model.addMVar(shape=m, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
    model.setObjective(0.5 * y @ P @ y + q @ y)
    constraint = model.addConstr(D @ y + s == b)
    i = 0
    for (dim, cone) in cones:
        cone = str(cone)
        if constants.ZERO_CONE in cone:
            model.addConstr(s[i:i + dim] == 0)
        elif constants.NONNEGATIVE_CONE in cone:
            model.addConstr(s[i:i + dim] >= 0)
        elif constants.SECOND_ORDER_CONE in cone:
            model.setParam("QCPDual", 1)
            model.addConstr(s[i] >= 0)
            norm_squared = gp.quicksum([s[j] ** 2 for j in range(i + 1, i + dim)])
            model.addConstr(norm_squared <= s[i] ** 2)
        i += dim
    model.optimize()

    status = model.Status
    if status in SOLVED_STATUS:
        optimal_value = model.ObjVal
        optimal_solution = y.X
        primal_slacks = s.X
        dual_solution = -constraint.Pi
        solve_time = model.Runtime
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, status)
    return (None, None, None, None, constants.TIME_LIMIT, status)