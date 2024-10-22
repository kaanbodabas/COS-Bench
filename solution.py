class Solution:
    def __init__(self, solver, optimal_value, optimal_solution, dual_solution):
        self.solver = solver
        self.optimal_value = optimal_value
        self.optimal_solution = optimal_solution
        self.dual_solution = dual_solution

    def __str__(self):
        return f"""{self.solver} Solution\n
                   Optimal Value: {self.optimal_value}\n
                   Optimal Solution: {self.optimal_solution}\n
                   Dual Solution: {self.dual_solution}\n"""