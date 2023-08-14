#!/bin/bash
#SBATCH --output=../eo/MI05A.out
#SBATCH --error=../eo/MI05A.err
#SBATCH --parsable
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Joachim_Collin@hms.harvard.edu
#SBATCH -t 0-01:00:00

set -e
module load gcc/6.2.0
module load miniconda3/4.10.3
module load cuda/11.2

source activate EV1
python -u ../scripts/MI05A_Ensembles_Predictions.py && echo "PYTHON SCRIPT COMPLETED"