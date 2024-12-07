from enums import Solver
from scipy import sparse
import numpy as np

def send_triu_vec_to_tril_vec(n, d, solver):
    upper_rows, upper_cols = np.triu_indices(d)
    lower_rows, lower_cols = np.tril_indices(d)
    X = np.zeros((n, n))
    k = 0
    for i, j in zip(upper_rows, upper_cols):
        x = 1
        if i != j and solver == Solver.MOSEK:
            x = np.sqrt(2)
        X[k][np.where((lower_rows == j) & (lower_cols == i))[0][0]] = x
        k += 1
    return sparse.csc_matrix(X)