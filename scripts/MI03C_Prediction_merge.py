import sys
from MI_Classes import PredictionsMerge

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('PRED-METRIC.BMI.2.2-Metric.age.2Metric.sexPrs.std.bmi')  # target
    sys.argv.append('test')  # fold
    
print(sys.argv)

# Compute results
Predictions_Merge = PredictionsMerge(target=sys.argv[1], fold=sys.argv[2])
Predictions_Merge.preprocessing()
Predictions_Merge.merge_predictions()
Predictions_Merge.postprocessing()
Predictions_Merge.save_merged_predictions()

# Exit
print('Done.')
sys.exit(0)