#!/bin/bash
#SBATCH --output=../eo/MI09.out
#SBATCH --error=../eo/MI09.err
#SBATCH --parsable
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Joachim_Collin@hms.harvard.edu
#SBATCH -t 0-10:00:00
#SBATCH --mem-per-cpu=8GB


set -e
module load gcc/6.2.0
module load miniconda3/4.10.3
module load cuda/11.2
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/n/app/cuda/11.2/ 
source activate EV1

python -u ../scripts/MI09_VAE.py && echo "PYTHON SCRIPT COMPLETED"
