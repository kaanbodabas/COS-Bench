from constants import NUM_CORES, TIME_LIMIT
from joblib import Parallel, delayed
from utils import verify, compute
from tqdm import tqdm
import pandas as pd
import os

def check_optimality(problem, solver, eps, solution):
    success = True
    if not verify.is_solution_optimal(problem, solver, eps):
        print(f"Solver {solver} reports an inaccurate primal-dual solution!")
        solution.solve_time = TIME_LIMIT
        success = False
    return {"Solver": solver, "Solve Time": solution.solve_time, "Success": success}

def start(solvers, csv_filename, problem_class, problem_data, eps=(10**-3, 10**-3, 10**-3), verbose=False):
    solutions = []
    for i, solver in enumerate(solvers):
        print(f"Solving instances in {solver} ({i + 1}/{len(solvers)})")
        # print()
        
        def parse_instances(instance):
            problem = problem_class(*instance)
            problem.canonicalize()
            solution = problem.solve(solver, verbose)

            # optional to confirm valid reformulation
            # original_cvxpy_solution = problem.solve_original_in_cvxpy(verbose)
            # print("Original:", original_cvxpy_solution.optimal_value)
            # print("Reformulated:", solution.optimal_value + solution.constant_objective)

            return check_optimality(problem, solver, eps, solution)

        num_jobs = min(len(problem_data[0]), NUM_CORES)
        results = Parallel(num_jobs)(delayed(parse_instances)(instance) for instance in tqdm(zip(*problem_data)))
        solutions.extend(results)

    if not os.path.exists("output"):
        os.makedirs("output")
    pd.DataFrame(solutions, dtype=object).to_csv(f"output/{csv_filename}.csv")

def results(csv_filename, solvers, num_instances, plot_title):
    df = pd.read_csv(f"output/{csv_filename}.csv")
    solutions_df = df.drop(df.columns[0], axis=1)

    compute.plot_normalized_geometric_means(solvers, solutions_df, plot_title)
    compute.plot_performance_profiles(solvers, solutions_df, num_instances, plot_title)
    compute.plot_failure_rates(solvers, solutions_df, num_instances, plot_title)
    compute.plot_average_solve_times(solvers, solutions_df, num_instances, plot_title)