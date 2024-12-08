from problems.image_deblurring import ImageDeblurring
from utils import data, run
import numpy as np
import maps

solvers = maps.get_solvers("QP")
csv_filename = "image_deblurring"
num_instances = 2
blur_matrices = [data.get_2D_blur_matrix(28, 28, 8)] * num_instances
images = data.get_emnist_training_images(num_instances)
rhos = np.logspace(-4, 3, num_instances)
plot_title = "Image Deblurring"

run.start(solvers, csv_filename, ImageDeblurring, (blur_matrices, images, rhos))
run.results(csv_filename, solvers, num_instances, plot_title)