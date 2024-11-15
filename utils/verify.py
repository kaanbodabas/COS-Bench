from scipy import linalg as la
import numpy as np

def is_qp_solution_optimal(qp, solver, eps):
    solution = qp.get_solution(solver)
    y = np.array(solution.optimal_solution)
    s = np.array(solution.primal_slacks)
    z = np.array(solution.dual_solution)
    eps_primal, eps_dual, eps_duality_gap = eps

    b_inf_norm = la.norm(qp.b, np.inf)
    q_inf_norm = la.norm(qp.q, np.inf)
    y_norm = la.norm(y, 2)
    s_norm = la.norm(s, 2)
    z_norm = la.norm(z, 2)

    def is_primal_solution_feasible():
        primal_residual = qp.D @ y + s - qp.b
        relative_term = np.maximum(1, b_inf_norm + y_norm + s_norm)
        return la.norm(primal_residual, 2) < eps_primal * relative_term

    def is_dual_solution_feasible():
        dual_residual = qp.P @ y + qp.q + qp.D.T @ z
        relative_term = np.maximum(1, q_inf_norm + y_norm + z_norm)
        return la.norm(dual_residual, 2) < eps_dual * relative_term

    def is_duality_gap_small():
        primal_objective = solution.optimal_value
        dual_objective = -0.5 * y.T @ qp.P @ y - qp.b.T @ z
        relative_term = np.maximum(1, np.minimum(np.abs(primal_objective), np.abs(dual_objective)))
        return np.abs(primal_objective - dual_objective) < eps_duality_gap * relative_term

    return is_primal_solution_feasible() and is_dual_solution_feasible() and is_duality_gap_small()