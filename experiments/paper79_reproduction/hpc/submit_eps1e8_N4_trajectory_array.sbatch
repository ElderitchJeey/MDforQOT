#!/bin/bash -l
#SBATCH --job-name=qot_e8traj
#SBATCH --output=results/hpc_eps1e8_N4_trajectory/logs/%x_%A_%a.out
#SBATCH --error=results/hpc_eps1e8_N4_trajectory/logs/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/hpc_eps1e8_N4_trajectory/logs

source /opt/ohpc/pub/python/anaconda/3.8/etc/profile.d/conda.sh
conda activate mdforqot

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

WARM_TOLS=(1e-2 1e-3)
WARM_LABELS=(1em02 1em03)

WARM_TOL="${WARM_TOLS[$SLURM_ARRAY_TASK_ID]}"
WARM_LABEL="${WARM_LABELS[$SLURM_ARRAY_TASK_ID]}"
OUT="results/hpc_eps1e8_N4_trajectory/ising_mixed_N4_eps1em08_warmtol${WARM_LABEL}_trajectory.csv"

python -m experiments.paper79_reproduction.run_paper79_style_trajectory \
  --kind ising \
  --N 4 \
  --case mixed \
  --seed 0 \
  --eps 1e-8 \
  --eps0 1 \
  --q 10 \
  --T 200000 \
  --max_gibbs_calls 200000 \
  --record_every 1000 \
  --modes cold,warm \
  --M_list 1,2,5 \
  --tol_tr 1e-4 \
  --tol_F 1e-8 \
  --tol_inner 1e-4 \
  --warm_tol "$WARM_TOL" \
  --out "$OUT"
