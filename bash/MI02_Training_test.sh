#!/bin/bash
#SBATCH --parsable
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=truncate



module load gcc/6.2.0
module load miniconda3/4.10.3
module load cuda/11.2
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/n/app/cuda/11.2/
source activate EV1




srun -n 1 -t "$((${14}-3))" --mem "$(($SLURM_MEM_PER_CPU-1))" bash -c "{ python -u ../scripts/MI02_Training.py  $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}; } && touch ../eo/version/$SLURM_JOB_NAME.success"
sleep 5 # wait slurm get the job status into its database
echo Summary:
sacct --format=JobID,Submit,Start,End,State,Partition,ReqTRES%30,CPUTime,MaxRSS,NodeList%30 --units=M -j $SLURM_JOBID
