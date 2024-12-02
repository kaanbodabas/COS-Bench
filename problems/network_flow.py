

class NetworkFlow:
    def __init__(self, A, b, c, u):
        self.A = A
        self.b = b
        self.c = c
        self.u = u

        self.n = len(self.c)

        self.P = None
        self.q = None
        self.D = None
        self.b = None
        self.cones = None

        self.original_cvxpy_solution = None
        self.solutions = {}

    def solve_original_in_cvxpy(self, verbose=False):
        pass

    def get_original_cvxpy_solution(self):
        pass
    
    def canonicalize(self):
        pass

    def solve(self, solver, verbose=False):
        pass

    def get_solution(self, solver):
        pass