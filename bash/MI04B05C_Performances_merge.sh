#!/bin/bash
#SBATCH --output=../eo/MI04B05C.out
#SBATCH --error=../eo/MI04B05C.err
#SBATCH --parsable
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Joachim_Collin@hms.harvard.edu
#SBATCH -t 0-00:20:00

set -e
module load gcc/6.2.0
module load miniconda3/4.10.3
module load cuda/11.2

source activate EV1

python -u ../scripts/MI04B05C_Performances_merge.py $1 $2 $3 && echo "PYTHON SCRIPT COMPLETED"