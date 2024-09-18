from problems_cvxpy.image_deblurring_cvxpy import ImageDeblurringCVXPY
from problems_clarabel.image_deblurring.image_deblurring_clarabel import ImageDeblurringClarabel
import numpy as np
import cvxpy as cp

class ImageDeblurringTest:
    def __init__(self, A, x, rho):
        self.in_cvxpy = ImageDeblurringCVXPY(A, x, rho)
        self.in_clarabel = ImageDeblurringClarabel(A, x, rho)

    def solve_both(self):
        self.in_cvxpy.solve()
        self.in_clarabel.solve()

    def compare_optimal_values(self):
        return self.in_cvxpy.optimal_value, self.in_clarabel.optimal_value

    def compare_original_images(self):
        return self.in_cvxpy.original_image, self.in_clarabel.original_image

if __name__ == "__main__":
    A = np.array([[2, 0], [0, -1]])
    x = np.array([1, 0])
    rho = 3
    test = ImageDeblurringTest(A, x, rho)
    test.solve_both()
    print(test.compare_optimal_values(), test.compare_original_images())