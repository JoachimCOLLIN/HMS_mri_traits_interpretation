import sys
from MI_Classes import PerformancesTuning

# Default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('RES-METRIC.BMI.2.2-Metric.age.2Metric.sexPrs.std.bmi')  # PRS
    
print(sys.argv)

# Compute results
Performances_Tuning = PerformancesTuning(target=sys.argv[1])
Performances_Tuning.load_data()
Performances_Tuning.preprocess_data()
Performances_Tuning.select_models()
Performances_Tuning.save_data()

# Exit
print('Done.')
sys.exit(0)