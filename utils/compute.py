from matplotlib import pyplot as plt
import numpy as np

SGM = " Shifted Geometric Means"
PP = " Performance Profiles"

def get_solve_times(solver, solutions_df):
    return solutions_df[solutions_df["Solver"] == solver.value]["Solve Time"].to_numpy(dtype=float)

def shifted_geometric_mean(solve_times, shift=10):
    return np.exp(np.sum(np.log(np.maximum(1, solve_times + shift)) / len(solve_times))) - shift

def normalized_geometric_mean(means):
    min_time = min(means.values())
    for solver in means:
        means[solver] /= min_time
    return means

def plot_normalized_geometric_means(solvers, solutions_df, title):
    means = {}
    for solver in solvers:
        solve_times = get_solve_times(solver, solutions_df)
        means[solver] = shifted_geometric_mean(solve_times)
    means = normalized_geometric_mean(means)

    bars = plt.bar([key.value for key in means.keys()], means.values())
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(round(height, 2)), ha="center", va="bottom")
    plt.title(title + SGM)
    plt.show()
    
def performance_profiles(solvers, solver_times, num_problems, taus, num_taus):
    for p in range(num_problems):
        min_time = min([solver_times[solver][p] for solver in solvers])
        for solver in solvers:
            solver_times[solver][p] /= min_time

    profiles = {}
    for solver in solvers:
        profiles[solver] = np.zeros(num_taus)
        for t in range(num_taus):
            ratio_indicator = 0
            for p in range(num_problems):
                if solver_times[solver][p] <= taus[t]:
                    ratio_indicator += 1
            profiles[solver][t] = ratio_indicator / num_problems
    return profiles

def plot_performance_profiles(solvers, solutions_df, num_problems, title, num_taus=1000):
    solver_times = {}
    taus = np.logspace(0, 4, num_taus)
    for solver in solvers:
        solve_times = get_solve_times(solver, solutions_df)
        solver_times[solver] = solve_times
    profiles = performance_profiles(solvers, solver_times, num_problems, taus, num_taus)

    for solver in solvers:
        plt.plot(taus, profiles[solver], label=solver)
    plt.xlabel("Performance Ratio")
    plt.xscale("log")
    plt.ylabel("Ratio of Problems Solved")
    plt.title(title + PP)
    plt.legend()
    plt.grid()
    plt.show()
