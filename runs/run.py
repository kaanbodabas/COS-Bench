from problems.image_deblurring import ImageDeblurring
from enums import get_qp_solvers
from utils import data, verify
from tqdm import tqdm
import pandas as pd

def check_optimality(problem, solver, eps):
    if not verify.is_solution_optimal(problem, solver, eps):
        print(f"Solver {solver} reports an inaccurate primal-dual solution!")
        return False
    return True

# TODO: parallelize
# TODO: generalize
def run_image_deblurring(blur_matrix_infos, images, rhos, csv_filename, solvers=get_qp_solvers(), eps=(10**-3, 10**-3, 10**-3)):
    solutions = []
    
    loading_bar = tqdm(solvers)
    for solver in loading_bar:
        loading_bar.set_description(f"Solving instances in {solver}")

        for blur_matrix_info, x in zip(blur_matrix_infos, images):
            for rho in rhos:
                A = data.get_2D_blur_matrix(*blur_matrix_info)
                problem = ImageDeblurring(A, x, rho)
                problem.canonicalize()

                solution = problem.solve(solver)

                if check_optimality(problem, solver, eps):
                    solutions.append({"Solver": solver, "Solve Time": solution.solve_time})
                else:
                    # TODO: Handle when solvers fail - set a time limit
                    solutions.append({"Solver": solver, "Solve Time": "fail"})
        
    pd.DataFrame(solutions, dtype=object).to_csv(f"output/{csv_filename}.csv")