#!/bin/bash
#SBATCH --output=../eo/MI03A.out
#SBATCH --error=../eo/MI03A.err
#SBATCH --parsable
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Joachim_Collin@hms.harvard.edu
#SBATCH -t 0-01:00:00
#SBATCH --mem-per-cpu=10GB


set -e
module load gcc/6.2.0
module load miniconda3/4.10.3
module load cuda/11.2
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/n/app/cuda/11.2/ 
source activate EV1

python -u ../scripts/MI03A_Predictions_generate.py $1 $2 $3 $4 $5 $6 $7 $8 ${9} ${10} ${11} ${12} ${13} && echo "PYTHON SCRIPT COMPLETED"
