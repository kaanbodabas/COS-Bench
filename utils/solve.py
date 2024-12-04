from ortools.pdlp import solvers_pb2, solve_log_pb2
from ortools.pdlp.python import pdlp
from scipy import sparse
import gurobipy as gp
import cvxpy as cp
import numpy as np
import clarabel
import mosek
import osqp
import scs

ZERO_CONE = "ZeroConeT"
NONNEGATIVE_CONE = "NonnegativeConeT"
TIME_LIMIT = 600
SOLVED = "solution returned"
SOLUTION_RETURNED = ["optimal", "optimal_inaccurate",
                     "Solved", "AlmostSolved",
                     gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL,
                     mosek.solsta.optimal, mosek.solsta.integer_optimal,
                     "solved", "solved inaccurate",
                     solve_log_pb2.TerminationReason.TERMINATION_REASON_OPTIMAL]

def parse_cones(cones):
    m = 0
    cone_infos = []
    for cone in cones:
        name = str(cone)
        m += cone.dim
        if name == ZERO_CONE + f"({cone.dim})":
            cone_infos.append((cone.dim, ZERO_CONE)) 
        elif name == NONNEGATIVE_CONE + f"({cone.dim})":
            cone_infos.append((cone.dim, NONNEGATIVE_CONE))
        else:
            raise Exception(f"Cone {cone} not supported!")
    return m, cone_infos

def with_cvxpy(n, P, q, D, b, cones, verbose):
    y = cp.Variable(n)
    m, cone_infos = parse_cones(cones) 
    s = cp.Variable(m)
    objective = 0.5 * cp.quad_form(y, cp.psd_wrap(P)) + q @ y
    constraints = [D @ y + s == b]
    for (dim, cone) in cone_infos:
        for i in range(dim):
            if cone == ZERO_CONE:
                constraints += [s[i] == 0]
            elif cone == NONNEGATIVE_CONE:
                constraints += [s[i] >= 0]
    problem = cp.Problem(cp.Minimize(objective), constraints)

    optimal_value = problem.solve(verbose=verbose)
    status = problem.status
    if status in SOLUTION_RETURNED:
        optimal_solution = y.value
        primal_slacks = s.value
        dual_solution = constraints[0].dual_value
        solve_time = problem.solver_stats.solve_time
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, SOLVED)
    return (None, None, None, None, TIME_LIMIT, status)
    

def with_clarabel(n, P, q, D, b, cones, verbose):
    settings = clarabel.DefaultSettings()
    settings.verbose = verbose
    settings.time_limit = TIME_LIMIT
    problem = clarabel.DefaultSolver(P, q, D, b, cones, settings)
    solution = problem.solve()
    
    status = solution.status
    if status in SOLUTION_RETURNED:
        optimal_value = solution.obj_val
        optimal_solution = solution.x
        primal_slacks = solution.s
        dual_solution = solution.z
        solve_time = solution.solve_time
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, SOLVED)
    return (None, None, None, None, TIME_LIMIT, status)

def with_gurobi(n, P, q, D, b, cones, verbose):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", int(verbose))
    env.setParam("TimeLimit", TIME_LIMIT)
    env.start()
    model = gp.Model("qp", env)
    y = model.addMVar(shape=n, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
    m, cone_infos = parse_cones(cones) 
    s = model.addMVar(shape=m, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
    objective = 0.5 * y @ P @ y + q @ y
    model.setObjective(objective)
    constraint = model.addConstr(D @ y + s == b)
    for (dim, cone) in cone_infos:
        for i in range(dim):
            if cone == ZERO_CONE:
                model.addConstr(s[i] == 0)
            elif cone == NONNEGATIVE_CONE:
                model.addConstr(s[i] >= 0)
    model.optimize()

    status = model.Status
    if status in SOLUTION_RETURNED:
        optimal_value = model.ObjVal
        optimal_solution = y.X
        primal_slacks = s.X
        dual_solution = -constraint.Pi
        solve_time = model.Runtime
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, SOLVED)
    return (None, None, None, None, TIME_LIMIT, status)

def with_mosek(n, P, q, D, b, cones, verbose):
    env = mosek.Env()
    task = env.Task()
    task.putintparam(mosek.iparam.log, int(verbose))
    task.putdouparam(mosek.dparam.optimizer_max_time, TIME_LIMIT)
    m, cone_infos = parse_cones(cones)
    task.appendcons(m)
    task.appendvars(n + m)
    for j in range(n):
        task.putcj(j, q[j])
        task.putvarbound(j, mosek.boundkey.fr, -np.inf, np.inf)
    P = sparse.tril(P, format="coo")
    task.putqobj(P.row, P.col, P.data)
    stacked_D = sparse.hstack([D, sparse.identity(m)])
    task.putaijlist(*sparse.find(stacked_D))
    for j in range(m):
        task.putconbound(j, mosek.boundkey.fx, b[j], b[j])
    j = 0
    for (dim, cone) in cone_infos:
        for i in range(n + j, n + j + dim):
            if cone == ZERO_CONE:
                task.putvarbound(i, mosek.boundkey.fx, 0, 0)
            elif cone == NONNEGATIVE_CONE:
                task.putvarbound(i, mosek.boundkey.lo, 0, np.inf)
        j += dim
    task.putobjsense(mosek.objsense.minimize)
    task.optimize()

    status = task.getsolsta(mosek.soltype.itr)
    if status in SOLUTION_RETURNED:
        optimal_value = task.getprimalobj(mosek.soltype.itr)
        primal_variables = np.array(task.getxx(mosek.soltype.itr))
        optimal_solution = primal_variables[:n]
        primal_slacks = primal_variables[n:]
        dual_solution = -np.array(task.gety(mosek.soltype.itr)[:m])
        solve_time = task.getdouinf(mosek.dinfitem.optimizer_time)
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, SOLVED)
    return (None, None, None, None, TIME_LIMIT, status)

def with_osqp(n, P, q, D, b, cones, verbose):
    m, cone_infos = parse_cones(cones)
    stacked_P = sparse.block_diag([P, np.zeros((m, m))], format="csc")
    stacked_q = np.hstack([q, np.zeros(m)])
    stacked_D = sparse.vstack([
        sparse.hstack([D, sparse.identity(m)]),
        sparse.hstack([np.zeros((m, n)), sparse.identity(m)])], format="csc")
    cone_lb = []
    cone_ub = []
    for (dim, cone) in cone_infos:
        for _ in range(dim):
            if cone == ZERO_CONE:
                cone_lb.append(0)
                cone_ub.append(0)
            elif cone == NONNEGATIVE_CONE:
                cone_lb.append(0)
                cone_ub.append(np.inf)
    lb = np.hstack([b, cone_lb])
    ub = np.hstack([b, cone_ub])
    problem = osqp.OSQP()
    problem.setup(stacked_P, stacked_q, stacked_D, lb, ub,
                  verbose=verbose, time_limit=TIME_LIMIT)
    solution = problem.solve()

    status = solution.info.status
    if status in SOLUTION_RETURNED:
        optimal_value = solution.info.obj_val
        optimal_solution = solution.x[:n]
        primal_slacks = solution.x[n:]
        dual_solution = solution.y[:m]
        solve_time = solution.info.run_time
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, SOLVED)
    return (None, None, None, None, TIME_LIMIT, status)

def with_pdlp(n, P, q, D, b, cones, verbose):
    problem = pdlp.QuadraticProgram()
    m, cone_infos = parse_cones(cones)
    problem.objective_vector = np.hstack([q, np.zeros(m)])
    problem.variable_lower_bounds = -np.inf * np.ones(n + m)
    problem.variable_upper_bounds = np.inf * np.ones(n + m)
    problem.constraint_matrix = sparse.vstack([
        sparse.hstack([D, sparse.identity(m)]),
        sparse.hstack([np.zeros((m, n)), sparse.identity(m)])], format="csc")
    cone_lb = []
    cone_ub = []
    for (dim, cone) in cone_infos:
        for _ in range(dim):
            if cone == ZERO_CONE:
                cone_lb.append(0)
                cone_ub.append(0)
            elif cone == NONNEGATIVE_CONE:
                cone_lb.append(0)
                cone_ub.append(np.inf)
    lb = np.hstack([b, cone_lb])
    ub = np.hstack([b, cone_ub])
    problem.constraint_lower_bounds = lb
    problem.constraint_upper_bounds = ub
    settings = solvers_pb2.PrimalDualHybridGradientParams()
    settings.verbosity_level = int(verbose)
    settings.termination_criteria.time_sec_limit = TIME_LIMIT
    solution = pdlp.primal_dual_hybrid_gradient(problem, settings)
    
    status = solution.solve_log.termination_reason
    if status in SOLUTION_RETURNED:
        optimal_value = solution.solve_log.solution_stats.convergence_information[0].primal_objective
        optimal_solution = solution.primal_solution[:n]
        primal_slacks = solution.primal_solution[n:]
        dual_solution = -solution.dual_solution[:m]
        solve_time = solution.solve_log.solution_stats.cumulative_time_sec
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, SOLVED)
    return (None, None, None, None, TIME_LIMIT, status)

def with_scs(n, P, q, D, b, cones, verbose):
    m, cone_infos = parse_cones(cones)
    stacked_P = sparse.block_diag([P, np.zeros((m, m))], format="csc")
    stacked_q = np.hstack([q, np.zeros(m)])
    stacked_D = sparse.vstack([
        sparse.hstack([D, sparse.identity(m)]), 
        sparse.hstack([np.zeros((m, n)), -sparse.identity(m)])], format="csc")
    z = 0
    l = 0
    for (dim, cone) in cone_infos:
        if cone == ZERO_CONE:
            z += dim
        elif cone == NONNEGATIVE_CONE:
            l += dim
    ub = np.hstack([b, np.zeros(m)])
    data = dict(P=stacked_P, A=stacked_D, b=ub, c=stacked_q)
    cone = dict(z=m + z, l=l)
    problem = scs.SCS(data, cone, verbose=verbose, time_limit_secs=TIME_LIMIT)
    solution = problem.solve()

    status = solution["info"]["status"]
    if status in SOLUTION_RETURNED:
        optimal_value = solution["info"]["pobj"]
        optimal_solution = solution["x"][:n]
        primal_slacks = solution["x"][n:]
        dual_solution = solution["y"][:m]
        solve_time = solution["info"]["solve_time"] / 1000
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, SOLVED)
    return (None, None, None, None, TIME_LIMIT, status)

# def with_sdpa(n, P, q, D, b, cones, verbose):

#     return (optimal_value, optimal_solution, primal_slacks,
#             dual_solution, solve_time, status)

# def with_copt(n, P, q, D, b, cones, verbose):

#     return (optimal_value, optimal_solution, primal_slacks,
#             dual_solution, solve_time, status)