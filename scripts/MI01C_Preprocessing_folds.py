import sys
from MI_Classes import PreprocessingFolds

# Options
# Regenerate the folds even if they already exist.
regenerate_data = True

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    # target
    sys.argv.append('METRIC.BMI.2.2')
    # organ
    sys.argv.append('Musculoskeletal') 
    
print(sys.argv)
target = sys.argv[1]

# Compute results
Preprocessing_Folds = PreprocessingFolds(target=target, organ=sys.argv[2],regenerate_data=regenerate_data)
Preprocessing_Folds.generate_folds()

# Exit
print('Done.')
sys.exit(0)
