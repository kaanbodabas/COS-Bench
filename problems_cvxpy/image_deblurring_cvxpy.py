import cvxpy as cp

class ImageDeblurringCVXPY:
    def __init__(self, A, x, rho):
        self.y = cp.Variable(len(x))
        self.objective = cp.norm(A @ self.y - x, 2)**2 + rho * cp.norm(self.y, 1)
        self.constraints = [self.y >= 0, self.y <= 1]
        self.problem = cp.Problem(cp.Minimize(self.objective), self.constraints)
        self.optimal_value = None
        self.original_image = None

    def solve(self):
        self.optimal_value = self.problem.solve()
        self.original_image = self.y.value

    def optimal_value(self):
        return self.optimal_value

    def original_image(self):
        return self.original_image