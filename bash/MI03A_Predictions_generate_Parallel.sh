#!/bin/bash
#SBATCH --output=../eo/MI03A_parallel.out
#SBATCH --error=../eo/MI03A_parallel.err
#SBATCH -t 0-00:10:00
#SBATCH --parsable
#SBATCH --open-mode=truncate
#SBATCH -p priority

memory=8G

regenerate_predictions=true
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
outer_folds=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9")
folds=( "train" "val" "test" )



declare -a IDs=()

for target in "${targets[@]}"; do
    for organ in "${organs[@]}"; do
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
            views=( "FullBody" )
        elif [ $organ == "PhysicalActivity" ]; then
            views=( "FullWeek" )
        fi
        for view in "${views[@]}"; do
            if [ $organ == "Brain" ]; then
                transformations=( "SagittalRaw" "SagittalReference" "CoronalRaw" "CoronalReference" "TransverseRaw" "TransverseReference" )
            elif [ $organ == "Eyes" ]; then
                transformations=( "Raw" )
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
                    transformations=( "Mixed"  )
                fi
            elif [ $organ == "PhysicalActivity" ]; then
                transformations=( "GramianAngularField1minDifference" "GramianAngularField1minSummation" "MarkovTransitionField1min" "RecurrencePlots1min" )
            fi
            for transformation in "${transformations[@]}"; do
                for architecture in "${architectures[@]}"; do
                    for optimizer in "${optimizers[@]}"; do
                        for n_fc_layers in "${n_fc_layersS[@]}"; do
                            if [ $n_fc_layers == "0" ]; then
                                    n_fc_nodesS_amended=( "0" )
                            else
                                    n_fc_nodesS_amended=( "${n_fc_nodesS[@]}" )
                            fi
                            for n_fc_nodes in "${n_fc_nodesS_amended[@]}"; do
                                for learning_rate in "${learning_rates[@]}"; do
                                    for weight_decay in "${weight_decays[@]}"; do
                                        for dropout_rate in "${dropout_rates[@]}"; do
                                            for data_augmentation_factor in "${data_augmentation_factors[@]}"; do
                                                for outer_fold in "${outer_folds[@]}"; do
                                                    version=${target}_${organ}_${view}_${transformation}_${architecture}_${n_fc_layers}_${n_fc_nodes}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${data_augmentation_factor}_${outer_fold}
                                                    name=MI03A_$version
                                                    job_name="$name.job"
                                                    out_file="../eo/predictions/$name.out"
                                                    err_file="../eo/predictions/$name.err"
                                                    # time as a function of the dataset
                                                    if [ $organ == "Arterial" ]; then
                                                        time=10 # 9k samples
                                                    elif [ $organ == "Brain" ] || [ $organ == "Heart" ] || [ $organ == "Abdomen" ]; then
                                                        time=25 #45k samples
                                                    elif [ $organ == "Musculoskeletal" ]; then
                                                        time=60 #45k samples
                                                    elif [ $organ == "Eyes" ] || [ $organ == "PhysicalActivity" ]; then
                                                        time=150 #90-100k samples
                                                    fi
                                                    # double the time for datasets for which each image is available for both the left and the right side
                                                    if [ $organ == "Eyes" ] || [ $organ == "Arterial" ] || [ $view == "Hips" ] || [ $view == "Knees" ]; then
                                                        time=$(( 2*$time ))
                                                    fi
                                                    # time multiplicator as a function of architecture
                                                    if [ $architecture == "InceptionResNetV2" ] || [ $architecture == "ResNeXt50" ]  || [ $architecture == "ResNeXt101" ] || [ $architecture == "ResNet152V2" ]; then
                                                        time=$(( 2*$time ))
                                                    fi
                                                    # time multiplicator for failed jobs. default: 1
                                                    #time=$(( 2*$time ))
                                                    #check if all weights have already been generated. If not, do not run the model.
                                                    path_weights="../data/weights/model-weights_${version}.h5"
                                                    if ! test -f $path_weights; then
                                                        echo The weights at $path_weights cannot be found. The job cannot be run.
                                                        continue
                                                    fi
                                                    #if regenerate_predictions option is on or if one of the predictions is missing, run the job
                                                    to_run=false
                                                    for fold in "${folds[@]}"; do
                                                        path_predictions="../data/dataframes/Predictions_${target}_${organ}_${view}_${transformation}_${architecture}_${n_fc_layers}_${n_fc_nodes}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${data_augmentation_factor}_${fold}_${outer_fold}.csv"
                                                        if ! test -f $path_predictions; then
                                                            to_run=true
                                                        fi
                                                    done
                                                    if $regenerate_predictions; then
                                                        to_run=true
                                                    fi
                                                    echo $to_run
                                                    if $to_run; then
                                                        echo Submitting job for $version
                                                        ID=$(sbatch  --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI03A_Predictions_generate.sh $target $organ $view $transformation $architecture $n_fc_layers $n_fc_nodes $optimizer $learning_rate $weight_decay $dropout_rate $data_augmentation_factor $outer_fold)
                                                        IDs+=($ID)
                                                    #else
                                                    #    echo Predictions for $version have already been generated.
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