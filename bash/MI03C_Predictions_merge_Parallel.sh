#!/bin/bash
#SBATCH --output=../eo/MI03C_parallel.out
#SBATCH --error=../eo/MI03C_parallel.err
#SBATCH -t 0-01:00:00
#SBATCH --parsable
#SBATCH --open-mode=truncate
#SBATCH -p priority
#SBATCH --mail-user=Joachim_Collin@hms.harvard.edu


targets=( "METRIC.W.2" "METRIC.WC.2" "METRIC.HEIGHT.2" "METRIC.BMI.2.2")
folds=( "val" "test" )
echo targets
declare -a IDs=()
for fold in "${folds[@]}"; do
    if [ $fold == "train" ]; then
        time=800
        memory=128G
    else
        time=60
        memory=16G
    fi
    for target in "${targets[@]}"; do
        version=MI03C_${target}_${fold}
        job_name="$version.job"
        out_file="../eo/prediction_merge/$version.out"
        err_file="../eo/prediction_merge/$version.err"
        ID=$(sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI03C_Predictions_merge.sh $target $fold)
        IDs+=($ID)
    done
done
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies
