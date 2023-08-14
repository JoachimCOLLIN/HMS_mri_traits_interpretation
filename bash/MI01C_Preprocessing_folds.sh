#!/bin/bash
#SBATCH --output=../eo/MI01C.out
#SBATCH --error=../eo/MI01C.err
#SBATCH -t 15
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --parsable
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Joachim_Collin@hms.harvard.edu

set -e
module load gcc/6.2.0
module load miniconda3/4.10.3
source activate EV1

python -u ../scripts/MI01C_Preprocessing_folds.py $1 && echo "PYTHON SCRIPT COMPLETED"
