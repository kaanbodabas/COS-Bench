from scipy import linalg as la
import numpy as np
import emnist

def get_emnist_training_images(subset="letters"):
    images, _ = emnist.extract_training_samples(subset)
    column_stacked_images = [image.flatten(order="F") for image in images]
    return np.array(column_stacked_images[:100]) / 255 # temporarily small dataset

# https://github.com/stellatogrp/data_driven_optimizer_guarantees/blob/main/opt_guarantees/examples/mnist.py
def get_2D_blur_matrix(m, n, width):

    def get_blur_matrix(m, width):
        half_length = int(np.ceil((width-1)/2))    
        rows, cols = np.zeros(m), np.zeros(m)
        cols[:1 + half_length] = 1 / width
        rows[:1 + half_length] = 1 / width
        return la.toeplitz(cols, rows)

    blur_cols = get_blur_matrix(m, width)
    blur_rows = get_blur_matrix(n, width)
    return np.kron(blur_rows, blur_cols)