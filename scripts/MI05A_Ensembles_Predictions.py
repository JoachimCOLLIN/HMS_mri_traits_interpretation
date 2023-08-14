import sys
from MI_Classes import EnsemblesPredictions

regenerate_models=True
# Default parameters
if len(sys.argv) != 5:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('RES-METRIC.BMI.2.2-Metric.age.2Metric.sexPrs.std.bmi')  # target

print(sys.argv)

# Generate results
EP = EnsemblesPredictions(target=sys.argv[1], regenerate_models=regenerate_models)
EP.load_data()
EP.generate_ensemble_predictions()
EP.save_predictions()
# Exit
print('Done.')
sys.exit(0)