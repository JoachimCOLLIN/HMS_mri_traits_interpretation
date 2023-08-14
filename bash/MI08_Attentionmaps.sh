#!/bin/bash
#SBATCH --output=../eo/MI08.out
#SBATCH --error=../eo/MI08.err
#SBATCH --parsable
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Joachim_Collin@hms.harvard.edu
#SBATCH -t 0-05:00:00
#SBATCH --mem-per-cpu=64GB


set -e
module load gcc/6.2.0
module load miniconda3/4.10.3
module load cuda/11.2

source activate EV1

python -u ../scripts/MI08_Attentionmaps.py $1 $2 $3 $4 && echo "PYTHON SCRIPT COMPLETED"

