class Solution:
    def __init__(self, solver, constant_objective, optimal_value, optimal_solution,
                 primal_slacks, dual_solution, solve_time, status):
        self.solver = solver
        self.constant_objective = constant_objective
        self.optimal_value = optimal_value
        self.optimal_solution = optimal_solution
        self.primal_slacks = primal_slacks
        self.dual_solution = dual_solution
        self.solve_time = solve_time
        self.status = status

    def __str__(self):
        return f"""\n{self.solver} Solution\n
                   Optimal Value: {self.optimal_value + self.constant_objective}\n
                   Solve Time: {self.solve_time}\n
                   Status: {self.status}\n"""