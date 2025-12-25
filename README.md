# MDforQOT

This repository implements and empirically evaluates several algorithms for **entropic quantum optimal transport (QOT)**, with a focus on **convergence, stability, and limiting consistency** of the computed coupling.

The main goal is to compare different optimization schemes under controlled numerical settings and to understand their behavior across dimensions, number of marginals, and regularization strength.

---

## Implemented algorithms

The repository currently includes the following methods:

* **BGDA**
  Block Gradient Descent–Ascent scheme (baseline method from prior work)

* **Marginal KL-descent**
  Potential-based marginal KL descent on the Gibbs family (original)

* **Mirror-descent type quantum Sinkhorn**
  An MD interpretation of noncommutative Sinkhorn with inner iterations (original)

All methods operate on the same entropic QOT formulation and are evaluated using consistent stopping criteria.

---

## Repository structure

```text
MDforQOT/
│
├── src/
|   ├── linalg/tensor         # basic definitions
│   ├── SolverofEQOT.py       # Core solvers (BGDA, KL, MD-Sinkhorn)
│   ├── metrics.py            # Distances, convergence checks, same-limit tests
│   └── instances.py          # Random QOT instance generation
│
├── experiments/              # Small sanity / correctness tests
│   └── sweeps/               # Parameter sweeps and large-scale experiments
│
├── figures/                  # Generated figures (not tracked)
├── results/                  # CSV results from sweeps (not tracked)
└── README.md
```

---

## Main experimental questions

The experiments in this repository are designed to address the following questions:

1. **Convergence**
   Do different algorithms converge under the same stopping tolerance?

2. **Same-limit consistency**
   When multiple algorithms converge, do they converge to the *same limiting coupling* (up to numerical precision)?

3. **Computational cost**
   How do Gibbs calls and wall-clock time scale with:

   * local dimension ( d )
   * number of marginals ( N )
   * entropic regularization parameter ( \varepsilon )

4. **Algorithmic sensitivity**
   How sensitive are the algorithms to step sizes, inner iteration counts, and regularization strength?

---

## Reproducing the main experiments

All commands below should be run from the repository root.

### 1. Convergence and same-limit vs BGDA (fixed ( N=2 ))

Sweep over dimensions ( d ) and random seeds, comparing BGDA, KL-descent, and MD-Sinkhorn:

```bash
python -m experiments.sweeps.sweep_vs_bgda_N2
```

This generates CSV files in `results/`.

To plot convergence rates and same-limit consistency:

```bash
python -m experiments.sweeps.plot_convergence_rate_N2
python -m experiments.sweeps.plot_same_limit_consistency_N2
```

---

### 2. Scaling with dimension (speed comparison)

For fixed ( N=2 ), compare:

* convergence rate vs ( d )
* Gibbs calls vs ( d )
* wall-clock time vs ( d )
* time per Gibbs call vs ( d )

```bash
python -m experiments.sweeps.sweep_speed_vs_d_N2
python -m experiments.sweeps.plot_speed_vs_d_N2
```

BGDA step sizes are selected via grid search over a predefined range.

---

### 3. Dependence on entropic regularization ( \varepsilon )

Fix ( N=3, d=3 ), sweep ( \varepsilon \in [10^{-2}, 1] ), and compare:

* convergence rate
* median Gibbs calls (among converged runs)

```bash
python -m experiments.sweeps.sweep_eps_KL_vs_MD
python -m experiments.sweeps.plot_eps_KL_vs_MD
```

---

### 4. Inner iteration sanity checks (MD-Sinkhorn)

To verify convergence of the **inner mirror-descent loop** used in MD-Sinkhorn:

```bash
python -m experiments.test_inner_md_convergence
```

This confirms that the inner iteration reliably reduces marginal mismatch to the prescribed tolerance.

---

## Notes on parameters and reproducibility

* Random instances are generated using fixed seeds.
* Convergence is assessed using the trace-norm marginal error.
* BGDA step sizes are tuned over a predefined grid; KL and MD use fixed step sizes motivated by theory.
* All experiments are designed to be runnable on a personal machine for moderate dimensions; larger sweeps are intended for cluster execution.

---

## Status

The current focus of the repository is **empirical validation and comparison**.
Theoretical analysis and manuscript preparation are ongoing.
