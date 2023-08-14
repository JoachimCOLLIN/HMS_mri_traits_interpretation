#!/bin/bash
#SBATCH --output=../eo/MI04C.out
#SBATCH --error=../eo/MI04C.err
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
python -u ../scripts/MI04C_Performances_tuning.py $1 && echo "PYTHON SCRIPT COMPLETED"