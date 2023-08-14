import sys
from MI_Classes import PredictionsGenerate

# options
# debug mode
debug_mode = False
# save predictions
save_predictions = True
# Default parameters
if len(sys.argv) != 14:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('PRED-METRIC.BMI.2.2-Metric.age.2Metric.sexPrs.std.bmi')  # target
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
    sys.argv.append('5')  # outer_fold

print(sys.argv)
# Compute results
Predictions_Generate = PredictionsGenerate(target=sys.argv[1], organ=sys.argv[2],
                                           view=sys.argv[3], transformation=sys.argv[4],
                                           architecture=sys.argv[5], n_fc_layers=sys.argv[6], n_fc_nodes=sys.argv[7],
                                           optimizer=sys.argv[8], learning_rate=sys.argv[9],
                                           weight_decay=sys.argv[10], dropout_rate=sys.argv[11],
                                           data_augmentation_factor=sys.argv[12], outer_fold=sys.argv[13],
                                           debug_mode=debug_mode)
Predictions_Generate.generate_predictions()
if save_predictions:
    Predictions_Generate.save_predictions()

# Exit
# Predictions_Generate.clean_exit()
