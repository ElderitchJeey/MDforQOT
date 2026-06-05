import json
import argparse
from utils import solve_dual_mosek

from solvers import *
from test_instances import tests

subgrad_functions = [subgrad_tol3, subgrad_tol4, subgrad_tol5, subgrad_tol6]
lbfgs_functions = [lbfgs_tol3, lbfgs_tol4, lbfgs_tol5, lbfgs_tol6]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a subset of tests and dump to JSON"
    )
    parser.add_argument(
        "--index", type=int,
        help="(0-based) index of the test in all_tests to run"
    )
    args = parser.parse_args()

    test = tests[args.index]

    cost_matrix, ptraces, dims, system_parts = test
    duals_e, primal_e = solve_dual_mosek(cost_matrix, ptraces, max_iters=20000, verbose=True)
    ground_truth = get_dual_value(duals_e, ptraces)

    test_logs = {
        'ground_truth': float(ground_truth),
        'solution_rank': int(sum(jnp.linalg.eigvalsh(primal_e).round(4) != 0.))
    }

    subgrad_logs = []
    for subgrad_function in subgrad_functions:
        _ = subgrad_function(*test)
        log = subgrad_function(*test)
        subgrad_logs.append(log)
    test_logs['subgrad_logs'] = subgrad_logs

    lbfgs_logs = []
    for lbfgs_function in lbfgs_functions:
        _ = lbfgs_function(*test)
        log = lbfgs_function(*test)
        lbfgs_logs.append(log)
    test_logs['lbfgs_logs'] = lbfgs_logs

    with open(f'data/test{args.index}.json', 'w') as file:
        json.dump(test_logs, file)
