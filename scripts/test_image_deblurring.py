from matplotlib import pyplot as plt
from enums import get_solvers
from utils import compute, data
from runs import run
import pandas as pd

blur_matrix_infos = [(28, 28, 8) for _ in range(1)]
images = data.get_emnist_training_images()[:1]
rhos = [1]
csv_filename = "image_deblurring"
plot_title = "Image Deblurring Solve Times"

num_problems = len(images) * len(rhos)
solvers = get_solvers("QP")

run.image_deblurring(solvers, blur_matrix_infos, images, rhos, csv_filename)

df = pd.read_csv(f"output/{csv_filename}.csv")
solutions_df = df.drop(df.columns[0], axis=1)

compute.plot_normalized_geometric_means(solvers, solutions_df, plot_title)
compute.plot_performance_profiles(solvers, solutions_df, num_problems, plot_title)