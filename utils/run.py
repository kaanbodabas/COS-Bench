from problems.image_deblurring import ImageDeblurring
from problems.network_flow import NetworkFlow
from utils import verify, compute
from enums import Problem
from tqdm import tqdm
import pandas as pd

problem_map = {Problem.NETWORK_FLOW: NetworkFlow,
               Problem.IMAGE_DEBLURRING: ImageDeblurring}

def check_optimality(problem, solver, eps, solutions, solve_time):
    if verify.is_solution_optimal(problem, solver, eps):
        solutions.append({"Solver": solver, "Solve Time": solve_time})
    else:
        print(f"Solver {solver} reports an inaccurate primal-dual solution!")

        # TODO: Handle when solvers fail - set a time limit
        # TODO: Status maps

        solutions.append({"Solver": solver, "Solve Time": "fail"})

# TODO: parallelize
def start(solvers, csv_filename, problem_type, problem_data, eps=(10**-3, 10**-3, 10**-3)):
    solutions = []
    loading_bar = tqdm(solvers)
    for solver in loading_bar:
        loading_bar.set_description(f"Solving instances in {solver}")

        for instance in zip(*problem_data):
            problem_class = problem_map[problem_type]
            problem = problem_class(*instance)
            problem.canonicalize()
            solution = problem.solve(solver)

            # TODO: solve original in cvxpy objective check

            check_optimality(problem, solver, eps, solutions, solution.solve_time)

    pd.DataFrame(solutions, dtype=object).to_csv(f"output/{csv_filename}.csv")

def results(csv_filename, solvers, num_instances, plot_title):
    df = pd.read_csv(f"output/{csv_filename}.csv")
    solutions_df = df.drop(df.columns[0], axis=1)

    compute.plot_normalized_geometric_means(solvers, solutions_df, plot_title)
    compute.plot_performance_profiles(solvers, solutions_df, num_instances, plot_title)