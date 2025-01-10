# COS-Bench: A Convex Optimization Solver Benchmarking Framework

## Usage Guide

### Installations
You'll need Python and pip installed first. Then, run `pip install -r requirements.txt` inside your virtual environment.

### Benchmarking
Just run `python -m scripts.YOUR_SCRIPT_FILENAME_HERE`. Resulting plots go in the `output` folder.

### Adding a solver
Using any existing solver as a guide, add a new file in the `solvers` folder and inside it complete the `solve(n, m, P, q, D, b, cones, verbose)` function. Add your solver to `enums.py` and `maps.py`.

### Adding a problem
Using any existing problem as a guide, add a new file in the `problems` folder and inside it create a class that inherits from `problem.Instance` and contains functions `__init__`, `solve_original_in_cvxpy`, and `canonicalize`. Add a new testing script in the `scripts` folder, using any existing script as a guide.

### CVXPY
You can add CVXPY as a solver in `maps.py` to incorporate it in the benchmarks.

### Configuration
Adjust `NUM_CORES` and `TIME_LIMIT` in `constants.py` as needed.

### Paper
See the paper for more information on the framework.