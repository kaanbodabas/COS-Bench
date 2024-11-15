import numpy as np

def shifted_geometric_mean(solve_times, shift=10):
    return np.exp(np.sum(np.log(np.maximum(1, solve_times + shift)) / len(solve_times))) - shift

def normalized_geometric_mean(stats):
    fastest_time = min(stats.values())
    for solver in stats:
        stats[solver] /= fastest_time
    
# def performance_profile():