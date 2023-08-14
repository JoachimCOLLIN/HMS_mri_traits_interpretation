import sys
from MI_Classes import PerformancesGenerate

# Default parameters
if len(sys.argv) != 14:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('PRED-METRIC.BMI.2.2-Metric.age.2Metric.sexPrs.std.bmi')
    sys.argv.append('Musculoskeletal')  # organ
    sys.argv.append('FullBody')  # view
    sys.argv.append('Mixed')  # transformation
    sys.argv.append('EfficientNetB7')  # architecture
    sys.argv.append('1')  # n_fc_layers
    sys.argv.append('1024')  # n_fc_nodes
    sys.argv.append('Adam')  # optimizer
    sys.argv.append('0.0001')  # learning_rate
    sys.argv.append('0.0')  # weight decay
    sys.argv.append('0.25')  # dropout_rate
    sys.argv.append('1')  # data_augmentation_factor
    sys.argv.append('test')  # fold

# # Default parameters for ensemble models
# if len(sys.argv) != 14:
#     print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
#     sys.argv = ['']
#     sys.argv.append('PRED-METRIC.BMI.2.2-Metric.age.2Metric.sexPrs.std.bmi')  # target
#     sys.argv.append('*')  # organ
#     sys.argv.append('*')  # view
#     sys.argv.append('*')  # transformation
#     sys.argv.append('*')  # architecture
#     sys.argv.append('*')  # n_fc_layers
#     sys.argv.append('*')  # n_fc_nodes
#     sys.argv.append('*')  # optimizer
#     sys.argv.append('*')  # learning_rate
#     sys.argv.append('*')  # weight_decay
#     sys.argv.append('*')  # dropout_rate
#     sys.argv.append('*')  # data_augmentation_factor
#     sys.argv.append('test')  # fold


# Compute results
print(sys.argv, False)
Performances_Generate = PerformancesGenerate(target=sys.argv[1], organ=sys.argv[2],
                                             view=sys.argv[3], transformation=sys.argv[4],
                                             architecture=sys.argv[5], n_fc_layers=sys.argv[6], n_fc_nodes=sys.argv[7],
                                             optimizer=sys.argv[8], learning_rate=sys.argv[9],
                                             weight_decay=sys.argv[10], dropout_rate=sys.argv[11],
                                             data_augmentation_factor=sys.argv[12], fold=sys.argv[13], debug_mode=False)
Performances_Generate.preprocessing()
Performances_Generate.compute_performances()
Performances_Generate.save_performances()

# Exit
print('Done.')
sys.exit(0)