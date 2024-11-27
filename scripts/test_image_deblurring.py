from matplotlib import pyplot as plt
from enums import get_qp_solvers
from utils import compute, data
from runs import run
import pandas as pd

# TODO: 2 steps - solve and compute/display

blur_matrix_infos = [(28, 28, 8) for _ in range(10)]
images = data.get_emnist_training_images()[:10]
rhos = [1]
csv_filename = "image_deblurring"

run.run_image_deblurring(blur_matrix_infos, images, rhos, csv_filename)

df = pd.read_csv(f"output/{csv_filename}.csv")
solutions_df = df.drop(df.columns[0], axis=1)
stats = {}

for solver in get_qp_solvers():
    solve_times = solutions_df[solutions_df["Solver"] == solver.value]["Solve Time"].to_numpy(dtype=float)
    stats[solver] = compute.shifted_geometric_mean(solve_times)
    
compute.normalized_geometric_mean(stats)

bars = plt.bar([key.value for key in stats.keys()], stats.values())
for bar in bars:
  height = bar.get_height()
  plt.text(bar.get_x() + bar.get_width() / 2, height, str(round(height, 2)), ha="center", va="bottom")

plt.title("Image Deblurring Solve Times Shifted Geometric Means")
plt.show()