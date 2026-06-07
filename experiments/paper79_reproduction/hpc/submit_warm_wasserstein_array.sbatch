#!/bin/bash -l
#SBATCH --job-name=qot_wwass
#SBATCH --output=results/hpc_warm_wasserstein/logs/%x_%A_%a.out
#SBATCH --error=results/hpc_warm_wasserstein/logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/hpc_warm_wasserstein/logs

source /opt/ohpc/pub/python/anaconda/3.8/etc/profile.d/conda.sh
conda activate mdforqot

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -m experiments.paper79_reproduction.hpc.run_warm_wasserstein_task \
  --task_id "$SLURM_ARRAY_TASK_ID" \
  --cases WP,WM,WC,WG \
  --d 20 \
  --eps_final 1e-3 \
  --eps0 1 \
  --q 10 \
  --max_inner 100000
