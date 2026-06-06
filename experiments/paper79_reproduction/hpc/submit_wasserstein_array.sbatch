#!/bin/bash -l
#SBATCH --job-name=qot_wass
#SBATCH --output=results/hpc_wasserstein/logs/%x_%A_%a.out
#SBATCH --error=results/hpc_wasserstein/logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/hpc_wasserstein/logs

source /opt/ohpc/pub/python/anaconda/3.8/etc/profile.d/conda.sh
conda activate mdforqot

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -m experiments.paper79_reproduction.hpc.run_wasserstein_task \
  --task_id "$SLURM_ARRAY_TASK_ID" \
  --cases WP,WM,WC,WG \
  --d 20 \
  --method_group all \
  --T_kl 30000 \
  --T_md 30000 \
  --lbfgs_max_iter 10000
