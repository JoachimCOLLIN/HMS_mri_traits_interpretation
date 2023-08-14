import sys
from MI_Classes import PredictionsConcatenate

# options
# save predictions
save_predictions = True

# Default parameters
if len(sys.argv) != 13:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('PRED-METRIC.BMI.2.2-Metric.age.2Metric.sexPrs.std.bmi')
    sys.argv.append('Musculoskeletal')  # organ
    sys.argv.append('FullBody')  # view
    sys.argv.append('Mixed')  # transformation
    sys.argv.append('DenseNet121')  # architecture
    sys.argv.append('1')  # n_fc_layers
    sys.argv.append('1024')  # n_fc_nodes
    sys.argv.append('Adam')  # optimizer
    sys.argv.append('0.0001')  # learning_rate
    sys.argv.append('0.2')  # weight decay
    sys.argv.append('0.5')  # dropout_rate
    sys.argv.append('1')  # data_augmentation_factor

print(sys.argv)

# Compute results
Predictions_Concatenate = \
    PredictionsConcatenate(target=sys.argv[1], organ=sys.argv[2], view=sys.argv[3],
                           transformation=sys.argv[4], architecture=sys.argv[5], n_fc_layers=sys.argv[6],
                           n_fc_nodes=sys.argv[7], optimizer=sys.argv[8], learning_rate=sys.argv[9],
                           weight_decay=sys.argv[10], dropout_rate=sys.argv[11], data_augmentation_factor=sys.argv[12])
Predictions_Concatenate.concatenate_predictions()
if save_predictions:
    Predictions_Concatenate.save_predictions()

# Exit
print('Done.')
sys.exit(0)