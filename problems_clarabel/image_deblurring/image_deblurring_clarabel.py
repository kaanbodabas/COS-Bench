from scipy import sparse
import numpy as np
import clarabel

class ImageDeblurringClarabel:
    def __init__(self, A, x, rho):
        self.x = x

        self.P = sparse.csc_matrix(2 * (A.T @ A))
        self.P = sparse.triu(self.P).tocsc()
        
        n = len(x)
        self.q = -2 * (A.T @ x) + rho * np.ones(n)
        
        I_n = sparse.identity(n)
        self.D = sparse.vstack([I_n, -I_n]).tocsc()
        
        self.b = np.concatenate([np.ones(n), np.zeros(n)])
        
        self.s = [clarabel.NonnegativeConeT(2 * n)]

        self.optimal_value = None
        self.original_image = None

    def solve(self):
        settings = clarabel.DefaultSettings()
        solver = clarabel.DefaultSolver(self.P, self.q, self.D, self.b, self.s, settings)
        solution = solver.solve()
        self.optimal_value = solution.obj_val + self.x.T @ self.x
        self.original_image = solution.x

    def optimal_value(self):
        return self.optimal_value

    def original_image(self):
        return self.original_image