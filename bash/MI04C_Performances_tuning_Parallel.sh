#!/bin/bash
#SBATCH --output=../eo/MI04C_parallel.out
#SBATCH --error=../eo/MI04C_parallel.err
#SBATCH --parsable
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-user=Joachim_Collin@hms.harvard.edu
#SBATCH -t 0-00:20:00

targets=( "METRIC.W.2" "METRIC.WC.2" "METRIC.HEIGHT.2" "METRIC.BMI.2.2")
# memory=32G
# time=60
memory=8G
time=60
declare -a ID
s=()
for target in "${targets[@]}"; do
    version=MI04C_${target}
    job_name="$version.job"
    out_file="../eo/performance_tuning/$version.out"
    err_file="../eo/performance_tuning/$version.err"
    ID=$(sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI04C_Performances_tuning.sh $target)
    IDs+=($ID)
done
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies
