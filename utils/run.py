from joblib import Parallel, delayed
from utils import verify, compute
from constants import NUM_CORES
from tqdm import tqdm
import pandas as pd


def check_optimality(problem, solver, eps, solution):
    success = True
    if not verify.is_solution_optimal(problem, solver, eps):
        print(f"Solver {solver} reports an inaccurate primal-dual solution!")
        success = False
    return {"Solver": solver, "Solve Time": solution.solve_time, "Success": success}

def start(solvers, csv_filename, problem_class, problem_data, eps=(10**-3, 10**-3, 10**-3), verbose=False):
    solutions = []
    loading_bar = tqdm(solvers)
    for solver in loading_bar:
        loading_bar.set_description(f"Solving instances in {solver}")
        print()

        def parse_instances(instance):
            problem = problem_class(*instance)
            problem.canonicalize()
            solution = problem.solve(solver, verbose)

            # optional to confirm valid reformulation
            original_cvxpy_solution = problem.solve_original_in_cvxpy(verbose)
            print("Original:", original_cvxpy_solution.optimal_value)
            print("Reformulated:", solution.optimal_value + solution.constant_objective)

            return check_optimality(problem, solver, eps, solution)

        num_jobs = min(len(problem_data[0]), NUM_CORES)
        results = Parallel(num_jobs)(delayed(parse_instances)(instance) for instance in zip(*problem_data))
        solutions.extend(results)

    pd.DataFrame(solutions, dtype=object).to_csv(f"output/{csv_filename}.csv")

def results(csv_filename, solvers, num_instances, plot_title):
    df = pd.read_csv(f"output/{csv_filename}.csv")
    solutions_df = df.drop(df.columns[0], axis=1)

    compute.plot_normalized_geometric_means(solvers, solutions_df, plot_title)
    compute.plot_performance_profiles(solvers, solutions_df, num_instances, plot_title)
    compute.plot_failure_rates(solvers, solutions_df, num_instances, plot_title)
    compute.plot_average_solve_times(solvers, solutions_df, num_instances, plot_title)