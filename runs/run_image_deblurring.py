from problems.image_deblurring import ImageDeblurring
from collections import defaultdict
from utils import data, verify
from enums import Solver

QP_SOLVERS = [Solver.cvxpy, Solver.clarabel]

def run(blur_matrix_infos, images, rhos, solvers=QP_SOLVERS, eps=(10**-5, 10**-5, 10**-5)):
    # blur_matrix_infos is a set of tuples
    # images is a set of images ready to be passed to the problem
    # rhos is a set of rhos ready to be passed to the problem
    # this generates len(images) x len(rhos) many problem instances

    solutions = defaultdict(list)

    for blur_matrix_info, x in zip(blur_matrix_infos, images):
        for rho in rhos:
            A = data.get_2D_blur_matrix(*blur_matrix_info)
            problem = ImageDeblurring(A, x, rho)
            problem.canonicalize()

            for solver in solvers:
                solution = problem.solve(solver)

                if not verify.is_qp_solution_optimal(problem, solver, eps):
                    raise Exception(f"Solver {solver} reports an inaccurate primal-dual solution!")
                
                solutions[solver].append(solution)
        
    # export solution information to csv?
    # build df -> get csv

    # compute statistics

    # display statistics