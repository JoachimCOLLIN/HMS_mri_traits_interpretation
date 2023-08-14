#!/bin/bash
#SBATCH --output=../eo/MI04A05B_parallel.out
#SBATCH --error=../eo/MI04A05B_parallel.err
#SBATCH -t 0-00:10:00
#SBATCH --parsable
#SBATCH --open-mode=truncate
#SBATCH -p priority
#SBATCH --mail-user=Joachim_Collin@hms.harvard.edu


memory=2G

regenerate_performances=true
folds=( "val" "test" )
targets=( "METRIC.W.2" "METRIC.WC.2" "METRIC.HEIGHT.2" "METRIC.BMI.2.2")
organs=( "Musculoskeletal")
architectures=("ResNet50" "DenseNet121")
n_fc_layersS=("1")
n_fc_nodesS=( "1024" )
optimizers=( "Adam" )
learning_rates=( "0.0001" )
weight_decays=( "0.0")
dropout_rates=("0.25")
data_augmentation_factors=( "1" )
ensemble_models=false



declare -a IDs=()




for target in "${targets[@]}"; do 
    for organ in "${organs[@]}"; do
        echo $organ
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
                for architecture in "${architectures[@]}"; do
                  for n_fc_layers in "${n_fc_layersS[@]}"; do
                        for n_fc_nodes in "${n_fc_nodesS[@]}"; do
                            for optimizer in "${optimizers[@]}"; do
                                for learning_rate in "${learning_rates[@]}"; do
                                    for weight_decay in "${weight_decays[@]}"; do
                                        for dropout_rate in "${dropout_rates[@]}"; do
                                            for data_augmentation_factor in "${data_augmentation_factors[@]}"; do
                                                for fold in "${folds[@]}"; do 
                                                    if $ensemble_models; then
                                                        organ="*"
                                                        view="*"
                                                        transformation="*"
                                                    fi
                version=${target}_${organ}_${view}_${transformation}_${architecture}_${n_fc_layers}_${n_fc_nodes}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${data_augmentation_factor}_${fold}
                                                    name=MI04A-$version
                                                    job_name="$name.job"
                                                    out_file="../eo/performance_generate/$name.out"
                                                    err_file="../eo/performance_generate/$name.err"
                                                    time=90
                                                    time=20 #debug mode
                                                    #allocate more time for the training fold because of the larger sample size
                                                    if [ $fold = "train" ]; then
                                                        time=$(( 8*$time ))
                                                    fi
                                                    #check if the predictions have already been generated. If not, do not run the model.
                                                    if ! test -f "../data/dataframes/Predictions_${version}.csv"; then
                                                        echo The predictions at "../data/dataframes/Predictions_${version}.csv" cannot be found. The job cannot be run.
                                                        break
                                                    fi
                                                    #if regenerate_performances option is on or if the performances have not yet been generated, run the job
                                                    if ! test -f "../data/dataframes/Performances_${version}.csv" || $regenerate_performances; then
                                                        echo "Submitting job for ${version}"
                                                        ID=$(sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI04A05B_Performances_generate.sh "$target" "$organ" "$view" "$transformation" "$architecture" "$n_fc_layers" "$n_fc_nodes" "$optimizer" "$learning_rate" "$weight_decay" "$dropout_rate" "$data_augmentation_factor" "$fold")
                                                        IDs+=($ID)
                                                        #else
                                                        #    echo Performance for $version have already been generated.
                                                    fi
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done


# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies
