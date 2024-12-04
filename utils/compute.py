from matplotlib import pyplot as plt
import numpy as np

SHIFTED_GEOMETRIC_MEANS = " Shifted Geometric Means"
PERFORMANCE_PROFILES = " Performance Profiles"
FAILURE_RATES = " Failure Rates"
AVERAGE_SOLVE_TIMES = " Average Solve Time"

def get_solve_times(solver, solutions_df):
    return solutions_df[solutions_df["Solver"] == solver.value]["Solve Time"].to_numpy(dtype=float) + 0.001

def get_num_fails(solver, solutions_df):
    return solutions_df[solutions_df["Solver"] == solver.value]["Success"].eq(False).sum()

def shifted_geometric_mean(solve_times, shift=10):
    return np.exp(sum(np.log(np.maximum(1, solve_times + shift)) / len(solve_times))) - shift

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

    plt.figure(figsize=(8, 6))
    bars = plt.bar([key.value for key in means.keys()], means.values())
    for bar in bars:
        height = bar.get_height()
        if height >= 100000:
            mean = format(height, ".2e")
        else:
            mean = round(height, 2)
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(mean), ha="center", va="bottom")
    plt.title(title + SHIFTED_GEOMETRIC_MEANS)
    plt.show()
    
def performance_profiles(solvers, solver_times, num_instances, taus, num_taus):
    for p in range(num_instances):
        min_time = min([solver_times[solver][p] for solver in solvers])
        for solver in solvers:
            solver_times[solver][p] /= min_time

    profiles = {}
    for solver in solvers:
        profiles[solver] = np.zeros(num_taus)
        for t in range(num_taus):
            ratio_indicator = 0
            for p in range(num_instances):
                if solver_times[solver][p] <= taus[t]:
                    ratio_indicator += 1
            profiles[solver][t] = ratio_indicator / num_instances
    return profiles

def plot_performance_profiles(solvers, solutions_df, num_instances, title, num_taus=1000):
    solver_times = {}
    taus = np.logspace(0, 4, num_taus)
    for solver in solvers:
        solve_times = get_solve_times(solver, solutions_df)
        solver_times[solver] = solve_times
    profiles = performance_profiles(solvers, solver_times, num_instances, taus, num_taus)

    plt.figure(figsize=(8, 6))
    for solver in solvers:
        plt.plot(taus, profiles[solver], label=solver)
    plt.xlabel("Performance Ratio")
    plt.xscale("log")
    plt.ylabel("Ratio of Problems Solved")
    plt.title(title + PERFORMANCE_PROFILES)
    plt.legend()
    plt.grid()
    plt.show()

def plot_failure_rates(solvers, solutions_df, num_instances, title):
    failure_rates = {}
    for solver in solvers:
        failure_rates[solver] = get_num_fails(solver, solutions_df) / num_instances
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar([key.value for key in failure_rates.keys()], failure_rates.values())
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(round(height, 4)), ha="center", va="bottom")
    plt.ylabel("Percent Failed")
    plt.title(title + FAILURE_RATES)
    plt.show()

def plot_average_solve_times(solvers, solutions_df, num_instances, title):
    averages = {}
    for solver in solvers:
        solve_times = get_solve_times(solver, solutions_df)
        averages[solver] = sum(solve_times) / num_instances

    plt.figure(figsize=(8, 6))
    bars = plt.bar([key.value for key in averages.keys()], averages.values())
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(round(height, 2)), ha="center", va="bottom")
    plt.ylabel("Time in Seconds")
    plt.title(title + AVERAGE_SOLVE_TIMES)
    plt.show()