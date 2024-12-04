from problems.image_deblurring import ImageDeblurring
from problems.network_flow import NetworkFlow
from joblib import Parallel, delayed
from utils import verify, compute
from enums import Problem
from tqdm import tqdm
import pandas as pd

NUM_CORES = 8
PROBLEM_MAP = {Problem.NETWORK_FLOW: NetworkFlow,
               Problem.IMAGE_DEBLURRING: ImageDeblurring}

def check_optimality(problem, solver, eps, solution):
    if not verify.is_solution_optimal(problem, solver, eps):
        print(f"Solver {solver} reports an inaccurate primal-dual solution!")
    # TODO: failure rates
    return {"Solver": solver, "Solve Time": solution.solve_time}

def start(solvers, csv_filename, problem_type, problem_data, eps=(10**-3, 10**-3, 10**-3)):
    solutions = []
    loading_bar = tqdm(solvers)
    for solver in loading_bar:
        loading_bar.set_description(f"Solving instances in {solver}")

        def parse_instances(instance):
            problem_class = PROBLEM_MAP[problem_type]
            problem = problem_class(*instance)
            problem.canonicalize()
            solution = problem.solve(solver)

            # optional to confirm valid reformulation
            print(" Reformulated:", solution.optimal_value + solution.constant_objective)
            original_cvxpy_solution = problem.solve_original_in_cvxpy()
            print("Original:", original_cvxpy_solution.optimal_value)

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