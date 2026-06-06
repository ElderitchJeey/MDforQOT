#!/bin/bash -l
#SBATCH --job-name=qot_qmix67
#SBATCH --output=results/hpc_qubit_mixed_N6N7/logs/%x_%A_%a.out
#SBATCH --error=results/hpc_qubit_mixed_N6N7/logs/%x_%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/hpc_qubit_mixed_N6N7/logs

source /opt/ohpc/pub/python/anaconda/3.8/etc/profile.d/conda.sh
conda activate mdforqot

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -m experiments.paper79_reproduction.hpc.run_qubit_mixed_task \
  --task_id "$SLURM_ARRAY_TASK_ID" \
  --Ns 6,7 \
  --kinds ising,random \
  --seeds 0 \
  --method_group all \
  --T_kl 3000 \
  --T_md 3000 \
  --lbfgs_max_iter 1000 \
  --outdir results/hpc_qubit_mixed_N6N7
