#!/bin/bash
#SBATCH --output=../eo/MI04B05C_parallel.out
#SBATCH --error=../eo/MI04B05C_parallel.err
#SBATCH -t 0-00:10:00
#SBATCH --parsable
#SBATCH --open-mode=truncate
#SBATCH -p short
#SBATCH --mail-user=Joachim_Collin@hms.harvard.edu


targets=( "METRIC.W.2" "METRIC.WC.2" "METRIC.HEIGHT.2" "METRIC.BMI.2.2")
echo $targets
folds=( "val" "test" )
echo $1
#Ensure that the ensemble_model parameter was specified
if [[ ! ($1 == "True" || $1 == "False") ]]; then
    echo ERROR. Usage: ./MI04B05C_Performance_merge_parallel.sh ensemble_models    ensemble_models must be either False to generate performances for simple models \(04B\), or True to generate performances for ensemble models \(05C\)
    exit
fi
ensemble_models=$1
memory=8G
time=5

#loop through the jobs to submit
declare -a IDs=()
for target in "${targets[@]}"; do
    for fold in "${folds[@]}"; do
        version=MI04B05C_${target}_${fold}_$1
        job_name="$version.job"
        out_file="../eo/performance_merge/$version.out"
        err_file="../eo/performance_merge/$version.err"
        ID=$(sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI04B05C_Performances_merge.sh $target $fold $ensemble_models)
        IDs+=($ID)
    done
done
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies
