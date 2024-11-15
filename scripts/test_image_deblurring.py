from enums import get_qp_solvers
from utils import compute, data
from runs import qps
import pandas as pd
import numpy as np

blur_matrix_infos = [(28, 28, 8) for _ in range(10)]
images = data.get_emnist_training_images()[:10]
rhos = [1]
csv_filename = "test1"

qps.run_image_deblurring(blur_matrix_infos, images, rhos, csv_filename)

df = pd.read_csv(f"output/{csv_filename}.csv")
solutions_df = df.drop(df.columns[0], axis=1)
stats = {}

for solver in get_qp_solvers():
    solve_times = solutions_df[solutions_df["Solver"] == solver.value]["Solve Time"].to_numpy(dtype=float)
    stats[solver] = compute.shifted_geometric_mean(solve_times)
    
compute.normalized_geometric_mean(stats)

for solver in stats:
    print(f"Solver: {solver}, SGM: {stats[solver]}")