#!/bin/bash
#SBATCH --output=../eo/MI00A.out
#SBATCH --error=../eo/MI00A.err
#SBATCH --mem-per-cpu=8G 
#SBATCH -c 1
#SBATCH -t 15
#SBATCH --parsable 
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=Joachim_Collin@hms.harvard.edu

set -e
module load gcc/6.2.0
module load miniconda3/4.10.3
source activate EV1 

python -u ../scripts/MI00A_SidePredictors.py && echo "CREATING DATA COMPLETED"
