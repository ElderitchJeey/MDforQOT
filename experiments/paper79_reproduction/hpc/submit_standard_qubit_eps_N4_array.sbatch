#!/bin/bash -l
#SBATCH --job-name=qot_qepsN4
#SBATCH --output=results/hpc_standard_qubit_eps_N4/logs/%x_%A_%a.out
#SBATCH --error=results/hpc_standard_qubit_eps_N4/logs/%x_%A_%a.err
#SBATCH --array=0-7
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/hpc_standard_qubit_eps_N4/logs

source /opt/ohpc/pub/python/anaconda/3.8/etc/profile.d/conda.sh
conda activate mdforqot

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

COMMON_ARGS=(
  --task_id "$SLURM_ARRAY_TASK_ID"
  --N 4
  --kinds ising,random
  --eps_values 1e-3,1e-6,1e-9,1e-12
  --seeds 0
)

python -m experiments.paper79_reproduction.hpc.run_qubit_eps_task \
  "${COMMON_ARGS[@]}" \
  --method_group all \
  --T_kl 1000000 \
  --T_md 1000000 \
  --lbfgs_max_iter 100000 \
  --max_gibbs_calls 1000000 \
  --tol_tr 1e-4 \
  --tol_f 1e-8 \
  --outdir results/hpc_standard_qubit_eps_N4/cold

python -m experiments.paper79_reproduction.hpc.run_warm_qubit_eps_task \
  "${COMMON_ARGS[@]}" \
  --eps0 1 \
  --q 10 \
  --max_inner 1000000 \
  --max_gibbs_calls 1000000 \
  --inner_tol 1e-2 \
  --final_tol 1e-4 \
  --tol_tr 1e-4 \
  --tol_F 1e-8 \
  --outdir results/hpc_standard_qubit_eps_N4/warm
