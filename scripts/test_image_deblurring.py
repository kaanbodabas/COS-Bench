from problems.image_deblurring import ImageDeblurring
from utils import data, run, maps

solvers = maps.get_solvers("QP")
csv_filename = "image_deblurring"
blur_matrices = [data.get_2D_blur_matrix(28, 28, 8)] * 2
images = data.get_emnist_training_images(2)
rhos = data.get_rho_range(2, 1, 2)
num_instances = 2
plot_title = "Image Deblurring Solve Times"

run.start(solvers, csv_filename, ImageDeblurring, (blur_matrices, images, rhos))
run.results(csv_filename, solvers, num_instances, plot_title)