#!/bin/bash
#SBATCH --output=../eo/MI02_parallel.out
#SBATCH --error=../eo/MI02_parallel.err
#SBATCH -t 0-00:05:00
#SBATCH --parsable
#SBATCH --open-mode=truncate
#SBATCH -p short




memory=8G
time=600

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
                        views=( "FullBody" ) #"Spine" "Hips" "Knees" 
                elif [ $organ == "PhysicalActivity" ]; then
                        views=( "FullWeek" )
                else
                        views=( "MRI" )
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
                                        # transformations=( "Mixed" "Figure" "Skeleton" "Flesh" )
                                        transformations=( "Mixed" )
                                fi
                        elif [ $organ == "PhysicalActivity" ]; then
                                if [ $view == "FullWeek" ]; then
                                        transformations=( "GramianAngularField1minDifference" "GramianAngularField1minSummation" "MarkovTransitionField1min" "RecurrencePlots1min" )
                                fi
                        fi        
                        for transformation in "${transformations[@]}"; do
                                for architecture in "${architectures[@]}"; do
                                        for n_fc_layers in "${n_fc_layersS[@]}"; do
                                                if [ $n_fc_layers == "0" ]; then
                                                        n_fc_nodesS_amended=( "0" )
                                                else
                                                        n_fc_nodesS_amended=( "${n_fc_nodesS[@]}" )
                                                fi
                                                for n_fc_nodes in "${n_fc_nodesS_amended[@]}"; do
                                                        for optimizer in "${optimizers[@]}"; do
                                                                for learning_rate in "${learning_rates[@]}"; do
                                                                        for weight_decay in "${weight_decays[@]}"; do
                                                                                for dropout_rate in "${dropout_rates[@]}"; do
                                                                                        for outer_fold in "${outer_folds[@]}"; do
                                                                                                for data_augmentation_factor in "${data_augmentation_factors[@]}"; do
                                                                                                version=MI02_${target}_${organ}_${view}_${transformation}_${architecture}_${n_fc_layers}_${n_fc_nodes}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${data_augmentation_factor}_${outer_fold}
                                                                                                        job_name="$version"
                                                                                                        out_file="../eo/version/$version.out"
                                                                                                        err_file="../eo/version/$version.err"
                                                                                                        if [ $(sacct -u joc9411 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep $version | egrep 'PENDING|RUNNING' | wc -l) -eq 0 ] ; then
                                                                                                                echo SUBMITTING: $version
                                                                                                                sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI02_Training_test.sh $target $organ $view $transformation $architecture $n_fc_layers $n_fc_nodes $optimizer $learning_rate $weight_decay $dropout_rate $data_augmentation_factor $outer_fold $time
                                                                                                                #else
                                                                                                                #        echo "Pending/Running: $version (or similar model)"
                                                                                                                fi
                                                                                                        #else
                                                                                                        #        echo "Already converged: $version"
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