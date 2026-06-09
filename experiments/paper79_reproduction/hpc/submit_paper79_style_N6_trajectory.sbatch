#!/bin/bash -l
#SBATCH --job-name=qot_trajN6
#SBATCH --output=results/hpc_paper79_style_N6/logs/%x_%j.out
#SBATCH --error=results/hpc_paper79_style_N6/logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/hpc_paper79_style_N6/logs

source /opt/ohpc/pub/python/anaconda/3.8/etc/profile.d/conda.sh
conda activate mdforqot

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -m experiments.paper79_reproduction.run_paper79_style_trajectory \
  --kind ising \
  --N 6 \
  --case mixed \
  --seed 0 \
  --eps 1e-3 \
  --eps0 1 \
  --q 10 \
  --T 100000 \
  --max_gibbs_calls 100000 \
  --modes cold \
  --record_every 10 \
  --tol_tr 1e-4 \
  --tol_F 1e-8 \
  --tol_inner 1e-4 \
  --warm_tol 1e-2 \
  --out results/hpc_paper79_style_N6/ising_mixed_N6_eps1em03_trajectory.csv
