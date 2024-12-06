from ortools.pdlp import solvers_pb2, solve_log_pb2
from ortools.pdlp.python import pdlp
from scipy import sparse
from enums import Solver
import gurobipy as gp
import cvxpy as cp
import numpy as np
import clarabel
import mosek
import osqp
import scs

ZERO_CONE = "ZeroConeT"
NONNEGATIVE_CONE = "NonnegativeConeT"
SECOND_ORDER_CONE = "SecondOrderConeT"
PSD_TRIANGLE_CONE = "PyPSDTriangleConeT"
TIME_LIMIT = 600
SOLVED = "solution returned"
SOLUTION_RETURNED = ["optimal", "optimal_inaccurate",
                     "Solved", "AlmostSolved",
                     gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL,
                     mosek.solsta.optimal, mosek.solsta.integer_optimal,
                     "solved", "solved inaccurate",
                     solve_log_pb2.TerminationReason.TERMINATION_REASON_OPTIMAL]

# column-major order version of cp.vec_to_upper_tri
def vec_to_upper_tri(s, n, d):
    rows, cols = np.tril_indices(d)
    A_rows = rows + d * cols
    A_cols = np.arange(n)
    A_vals = np.ones(A_cols.size)
    A = sparse.csc_matrix((A_vals, (A_rows, A_cols)), shape=(d * d, n))
    return cp.reshape(A @ s, (d, d), order='F').T

def send_upper_tri_vec_to_lower_tri_vec(n, d, solver):
    upper_rows, upper_cols = np.triu_indices(d)
    lower_rows, lower_cols = np.tril_indices(d)
    X = np.zeros((n, n))
    k = 0
    for i, j in zip(upper_rows, upper_cols):
        x = 1
        if i != j and solver == Solver.MOSEK:
            x = np.sqrt(2)
        X[k][np.where((lower_rows == j) & (lower_cols == i))[0][0]] = x
        k += 1
    return sparse.csc_matrix(X)

def with_cvxpy(n, m, P, q, D, b, cones, verbose):
    y = cp.Variable(n) 
    s = cp.Variable(m)
    objective = 0.5 * cp.quad_form(y, cp.psd_wrap(P)) + q @ y
    constraints = [D @ y + s == b]
    i = 0
    for (dim, cone) in cones:
        cone = str(cone)
        if ZERO_CONE in cone:
            constraints.append(s[i:i + dim] == 0)
        elif NONNEGATIVE_CONE in cone:
            constraints.append(s[i:i + dim] >= 0)
        elif SECOND_ORDER_CONE in cone:
            # TODO
            constraints.append(cp.norm(s[i + 1:i + dim], 2) <= s[i])
        elif PSD_TRIANGLE_CONE in cone:
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
    if status in SOLUTION_RETURNED:
        optimal_solution = y.value
        primal_slacks = s.value
        dual_solution = constraints[0].dual_value
        solve_time = problem.solver_stats.solve_time
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, SOLVED)
    return (None, None, None, None, TIME_LIMIT, status)
    
def with_clarabel(n, m, P, q, D, b, cones, verbose):
    settings = clarabel.DefaultSettings()
    settings.verbose = verbose
    settings.time_limit = TIME_LIMIT
    problem = clarabel.DefaultSolver(P, q, D, b, [cone[1] for cone in cones], settings)
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

def with_gurobi(n, m, P, q, D, b, cones, verbose):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", int(verbose))
    env.setParam("TimeLimit", TIME_LIMIT)
    env.start()
    model = gp.Model("qp", env)
    y = model.addMVar(shape=n, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY) 
    s = model.addMVar(shape=m, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
    objective = 0.5 * y @ P @ y + q @ y
    model.setObjective(objective)
    constraint = model.addConstr(D @ y + s == b)
    i = 0
    for (dim, cone) in cones:
        cone = str(cone)
        if ZERO_CONE in cone:
            model.addConstr(s[i:i + dim] == 0)
        elif NONNEGATIVE_CONE in cone:
            model.addConstr(s[i:i + dim] >= 0)
        elif SECOND_ORDER_CONE in cone:
            # TODO
            model.addConstr(gp.norm(s[i + 1:i + dim], 2) <= s[i])
        i += dim
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

def with_mosek(n, m, P, q, D, b, cones, verbose):
    env = mosek.Env()
    task = env.Task()
    task.putintparam(mosek.iparam.log, int(verbose))
    task.putdouparam(mosek.dparam.optimizer_max_time, TIME_LIMIT)
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
    for (dim, cone) in cones:
        cone = str(cone)
        if ZERO_CONE in cone:
            task.putvarboundsliceconst(i, i + dim, mosek.boundkey.fx, 0, 0)
        elif NONNEGATIVE_CONE in cone:
            task.putvarboundsliceconst(i, i + dim, mosek.boundkey.lo, 0, np.inf)
        elif SECOND_ORDER_CONE in cone:
            # TODO
            domain = task.appendquadraticconedomain(dim)
        elif PSD_TRIANGLE_CONE in cone:
            vec_dim = int(dim * (dim + 1) / 2)
            task.putvarboundsliceconst(i, i + vec_dim, mosek.boundkey.fr, -np.inf, np.inf)
            task.appendafes(vec_dim)
            F = send_upper_tri_vec_to_lower_tri_vec(vec_dim, dim, Solver.MOSEK)
            task.putafefentrylist(*sparse.find(F))
            domain = task.appendsvecpsdconedomain(vec_dim)
            task.appendacc(domain, list(range(j, j + vec_dim)), None)
            j += vec_dim
            dim = vec_dim
        i += dim
    task.putobjsense(mosek.objsense.minimize)
    task.optimize()

    status = task.getsolsta(mosek.soltype.itr)
    if status in SOLUTION_RETURNED:
        optimal_value = task.getprimalobj(mosek.soltype.itr)
        optimal_solution = np.array(task.getxxslice(mosek.soltype.itr, 0, n))
        primal_slacks = np.array(task.getxxslice(mosek.soltype.itr, n, n + m))
        dual_solution = -np.array(task.getyslice(mosek.soltype.itr, 0, m))
        solve_time = task.getdouinf(mosek.dinfitem.optimizer_time)
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, SOLVED)
    return (None, None, None, None, TIME_LIMIT, status)

def with_osqp(n, m, P, q, D, b, cones, verbose):
    stacked_P = sparse.block_diag([P, np.zeros((m, m))], format="csc")
    stacked_q = np.hstack([q, np.zeros(m)])
    stacked_D = sparse.vstack([
        sparse.hstack([D, sparse.identity(m)]),
        sparse.hstack([np.zeros((m, n)), sparse.identity(m)])], format="csc")
    cone_lb = []
    cone_ub = []
    for (dim, cone) in cones:
        cone = str(cone)
        if ZERO_CONE in cone:
            cone_lb.extend(np.zeros(dim))
            cone_ub.extend(np.zeros(dim))
        elif NONNEGATIVE_CONE in cone:
            cone_lb.extend(np.zeros(dim))
            cone_ub.extend(np.inf * np.ones(dim))
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

def with_pdlp(n, m, P, q, D, b, cones, verbose):
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
        if ZERO_CONE in cone:
            cone_lb.extend(np.zeros(dim))
            cone_ub.extend(np.zeros(dim))
        elif NONNEGATIVE_CONE in cone:
            cone_lb.extend(np.zeros(dim))
            cone_ub.extend(np.inf * np.ones(dim))
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

def with_scs(n, m, P, q, D, b, cones, verbose):
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
        if ZERO_CONE in cone:
            z += dim
        elif NONNEGATIVE_CONE in cone:
            l += dim
        elif SECOND_ORDER_CONE in cone:
            # TODO
            q_array.append(dim)
        elif PSD_TRIANGLE_CONE in cone:
            vec_dim = int(dim * (dim + 1) / 2)
            stacked_D = stacked_D.tolil()
            stacked_D[i:i + vec_dim, j:j + vec_dim] = -send_upper_tri_vec_to_lower_tri_vec(vec_dim, dim, Solver.SCS)
            stacked_D = stacked_D.tocsc()
            s += dim
            dim = vec_dim
        i += dim
        j += dim
    ub = np.hstack([b, np.zeros(m)])
    data = dict(P=stacked_P, A=stacked_D, b=ub, c=stacked_q)
    cone = dict(z=m + z, l=l, q=q_array, s=s)
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