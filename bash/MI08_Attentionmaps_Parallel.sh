#!/bin/bash
#SBATCH --output=../eo/MI08_parallel.out
#SBATCH --error=../eo/MI08_parallel.err
#SBATCH --parsable
#SBATCH -p short
#SBATCH --open-mode=truncate
#SBATCH --mail-user=Joachim_Collin@hms.harvard.edu
#SBATCH -t 0-00:20:00

targets=( "PRED-METRIC.BMI.2.2-Metric.age.2Metric.sexPrs.std.bmi" "RES-METRIC.BMI.2.2-Metric.age.2Metric.sexPrs.std.bmi")
organs=( "Musculoskeletal" )
memory=32G
time=1200
declare -a IDs=()
for target in "${targets[@]}"; do
    if [ $organ == "Brain" ]; then
                views=( "MRI" )
            elif [ $organ == "Eyes" ]; then
                views=( "Fundus" "OCT" )
            elif [ $organ == "Arterial" ]; then
                views=( "Carotids" )
            elif [ $organ == "Heart" ]; then
                views=( "MRI" )
            elif [ $organ == "Abdomen" ]; then
                views=( "Liver" "Pancreas" )
            elif [ $organ == "Musculoskeletal" ]; then
                #"Spine" "Hips" "Knees" 
                views=( "FullBody" )
            elif [ $organ == "PhysicalActivity" ]; then
                views=( "FullWeek" )
            else
                echo "Organ $organ does not match any Images organs."
            fi
            for view in "${views[@]}"; do
                        if [ $organ == "Brain" ]; then
                            transformations=( "SagittalRaw" "SagittalReference" "CoronalRaw" "CoronalReference" "TransverseRaw" "TransverseReference" )
                        elif [ $organ == "Arterial" ]; then
                            transformations=( "Mixed" "LongAxis" "CIMT120" "CIMT150" "ShortAxis" )
                        elif [ $organ == "Heart" ]; then
                            transformations=( "2chambersRaw" "2chambersContrast" "3chambersRaw" "3chambersContrast" "4chambersRaw" "4chambersContrast" )
                        elif [ $organ == "Abdomen" ]; then
                            transformations=( "Raw" "Contrast" )
                        elif [ $organ == "Musculoskeletal" ]; then
                            if [ $view == "Spine" ]; then
                                transformations=( "Sagittal" "Coronal" )
                            elif [ $view == "Hips" ] || [ $view == "Knees" ]; then
                                transformations=( "MRI" )
                            elif [ $view == "FullBody" ]; then
                                # "Figure" "Skeleton" "Flesh"
                                transformations=( "Mixed" )
                            fi
                        elif [ $organ == "PhysicalActivity" ]; then
                                transformations=( "GramianAngularField1minDifference" "GramianAngularField1minSummation" "MarkovTransitionField1min" "RecurrencePlots1min" )
                        elif [ $organ == "Eyes" ] || [ $organ == "Spine" ] || [ $organ == "Hips" ] || [ $organ == "Knees" ] || [ $organ == "FullBody" ]; then
                            transformations=( "Raw" )
                        else
                            echo "Organ $organ does not match any Images organs."
                        fi
                        for transformation in "${transformations[@]}"; do
                                version=MI04C_${target}_${view}_${transformation}
                                job_name="$version.job"
                                out_file="../eo/performance_tuning/$version.out"
                                err_file="../eo/performance_tuning/$version.err"
                                ID=$(sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI08_Attentionmaps.sh $target $organ $view $transformation False)
                                IDs+=($ID)
                        done
            done
done
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies
