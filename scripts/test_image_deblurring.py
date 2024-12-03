from utils import data, run, maps
from enums import Problem

# ensure the length of all 3 inputs are equal and that the inputs
# are in proper order described by the class
solvers = maps.get_solvers("QP")
csv_filename = "image_deblurring"
blur_matrix = [data.get_2D_blur_matrix(28, 28, 8)] * 2
images = data.get_emnist_training_images(2)
rhos = [1, 1]
num_instances = 1
plot_title = "Image Deblurring Solve Times"

run.start(solvers, csv_filename, Problem.IMAGE_DEBLURRING, (blur_matrix, images, rhos))
run.results(csv_filename, solvers, num_instances, plot_title)