import constants
import clarabel

SOLVED_STATUS = [1, 4]

def solve(n, m, P, q, D, b, cones, verbose):
    settings = clarabel.DefaultSettings()
    settings.verbose = verbose
    settings.time_limit = constants.TIME_LIMIT
    problem = clarabel.DefaultSolver(P, q, D, b, [cone[1] for cone in cones], settings)
    solution = problem.solve()
    
    status = solution.status
    if status in SOLVED_STATUS:
        optimal_value = solution.obj_val
        optimal_solution = solution.x
        primal_slacks = solution.s
        dual_solution = solution.z
        solve_time = solution.solve_time
        return (optimal_value, optimal_solution, primal_slacks,
                dual_solution, solve_time, status)
    return (None, None, None, None, constants.TIME_LIMIT, status)