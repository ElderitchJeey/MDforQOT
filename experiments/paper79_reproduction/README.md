# Paper79 Benchmark Reproduction

This folder adapts the benchmark instances from
`externel/paper79_subgradient` to the repo-native entropic QOT solvers.

The goal is to replace the previous standalone numerical section with
experiments on the same instance families used in paper79:

- Ising spin systems: `IG`, `IE`, `IT`, `IM`
- Random Hamiltonians: `RG`, `RE`, `RT`, `RM`
- Quantum Wasserstein/channel examples: `WP`, `WM`, `WC`, `WG`

The first bridge script runs KL descent and MD-Sinkhorn on one paper79 instance:

```bash
python -m experiments.paper79_reproduction.run_repo_solvers --index 8 --eps 1e-2 --method all
```

The main comparison runner focuses on L-BFGS for the entropy-regularized
dual versus the repo-native methods:

```bash
python -m experiments.paper79_reproduction.run_lbgfs_vs_ours --experiment main --indices all
```

It runs:

- paper79 L-BFGS on the von Neumann entropy-regularized dual
- KL descent with `eta=eps/N`
- KL descent with `eta=eps`
- MD-Sinkhorn with `M_inner = 1, 2, 5`

Summary CSVs report hit columns for several tolerances by default:
`1e-3`, `1e-4`, and `1e-5`, such as `hit_F_le_1em04` and
`hit_tr_gibbs_le_1em04`. The MD inner loop uses `tol_inner=1e-4` by
default to avoid oversolving every coordinate update; pass a different
`--tol_inner` for stricter or looser inner solves.

KL descent supports two built-in step-length rules. By default, the benchmark
runners use `--eta_kl_rules eps_over_N,eps`, so both `eta=eps/N` and `eta=eps`
appear in the same summary table. An explicit numeric `--eta_kl` still
overrides this and runs one manual step length.

Two presets are available:

- `--experiment main`: moderate entropy values, by default `eps in {1e-2, 1e-3}`
- `--experiment stress`: small-epsilon path, by default `eps in {1e-4, 1e-8, 1e-12}`

For a cheap local pipeline check:

```bash
python -m experiments.paper79_reproduction.run_lbgfs_vs_ours --tiny_smoke --experiment main --main_eps 0.5 --T_kl 5 --T_md 5 --lbfgs_max_iter 5
```

For a small paper79-like qubit trend table:

```bash
python -m experiments.paper79_reproduction.run_small_qubit_trend --N 4 --eps 1e-3
```

For the current qubit mixed HPC grid, the four default tasks are
`N=4/8 x ising/random` with `seed=0`. Submit them as a SLURM array:

```bash
sbatch experiments/paper79_reproduction/hpc/submit_qubit_mixed_array.sbatch
```

The current HPC scripts are:

- `submit_qubit_mixed_array.sbatch`: `N=4,8`, `ising/random`, `mixed`, `eps=1e-3`
- `submit_qubit_mixed_N6N7_array.sbatch`: `N=6,7`, `ising/random`, `mixed`, `eps=1e-3`
- `submit_qubit_eps_array.sbatch`: `N=4`, `ising/random`, `mixed`, `eps in {1e-4,1e-8,1e-12}`
- `submit_wasserstein_array.sbatch`: Wasserstein/channel `WP,WM,WC,WG`, `d=20`, `eps=1e-3`

To inspect the task mapping locally:

```bash
python -m experiments.paper79_reproduction.hpc.run_qubit_mixed_task --list
```

For tighter wall-clock limits, submit separate arrays with
`--method_group lbfgs`, `--method_group kl`, `--method_group md1`,
`--method_group md2`, or `--method_group md5` in the sbatch file. To use
several seeds, change both the launcher `--seeds` argument and the SLURM
`--array` range; for example `--Ns 4,8 --kinds ising,random --seeds 0,1,2,3,4`
has 20 tasks, so use `#SBATCH --array=0-19`.

Merge completed task CSVs with:

```bash
python -m experiments.paper79_reproduction.hpc.merge_hpc_results --indir results/hpc_qubit_mixed --out results/hpc_qubit_mixed_summary.csv
```

For the paper79-style quantum Wasserstein/channel cases at a local truncation
dimension `d=20`:

```bash
python -m experiments.paper79_reproduction.run_wasserstein_trend --d 20 --cases WP,WM,WC,WG --eps 1e-3
```

This script recreates WP/WM/WC/WG with NumPy Fock-basis operators, so it does
not require QuTiP.

For trajectory data and Figure A-D:

```bash
python -m experiments.paper79_reproduction.run_small_qubit_trajectory --kind ising --N 4 --eps 1e-3 --force_full_budget
python -m experiments.paper79_reproduction.plot_small_qubit_figures --trajectory results/small_qubit_trajectory_eps1e-3.csv --summary results/small_qubit_trend_eps1e-3.csv
```

Notes:

- Paper79 benchmarks the unregularized QOT value using MOSEK SDP ground truth,
  subgradient ascent, and L-BFGS with a very small entropy parameter.
- The repo-native methods solve entropic QOT and require logarithms of the target
  marginals. Many paper79 marginals are rank deficient, so the bridge uses jitter
  regularization for numerical stability.
- The comparison runner reports final-coupling consistency against L-BFGS through
  `same_limit_to_lbfgs`, `dist_pi_to_lbfgs`, and `objective_gap_to_lbfgs`.
- Full-size paper79 instances such as `d=50` Wasserstein and `N=12` qubit systems
  can be expensive on a laptop. The default Wasserstein runner uses `d=20` as a
  local compromise; increase `--d` only after small iteration-budget checks.
