import sys
from MI_Classes import PreprocessingMain

# Default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('METRIC.BMI.2.2')
    
print(sys.argv)

# checking if has been corrected
target = sys.argv[1]

# Compute results
Preprocessing_Main = PreprocessingMain(target)
Preprocessing_Main.generate_data()
Preprocessing_Main.save_data()

# Exit
print('Done.')
sys.exit(0)
