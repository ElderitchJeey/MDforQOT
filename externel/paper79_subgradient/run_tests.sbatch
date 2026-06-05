#!/bin/bash -l
#SBATCH --job-name=qot_tests
#SBATCH --output=data/logs/test_%a.out
#SBATCH --error=data/logs/test_%a.err
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# 1) go back to your submission dir
cd "$SLURM_SUBMIT_DIR"

# 2) now `module` should exist
module load cuda/12.6
module load cudnn/8.6

# 3) activate your conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311

# 4) make your code importable
export PYTHONPATH="$PWD:$PYTHONPATH"

# 5) run the test
python run_test.py --index $SLURM_ARRAY_TASK_ID
