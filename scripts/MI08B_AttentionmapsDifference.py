import sys
from MI_Classes import AttentionMapsDifference

nb_eids=1000
# Default parameters
if len(sys.argv) != 6:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append(
        {'target':'PRED-METRIC.BMI.2.2-Metric.age.2Metric.sexPrs.std.bmi', 
         'sub_parameters':
             {'architecture':'ResNet50',
               'n_fc_layers':'1',
               'n_fc_nodes':'1024',
               'optimizer':'Adam',
               'learning_rate':'0.0001', 
               'weight_decay':'0.0', 
               'dropout_rate':'0.5',
               'data_augmentation_factor':'1'
              }
            }
        )  # target1
    sys.argv.append(
        {'target':'RES-METRIC.BMI.2.2-Metric.age.2Metric.sexPrs.std.bmi', 
         'sub_parameters':
            {'architecture':'ResNet50',
                   'n_fc_layers':'1',
                   'n_fc_nodes':'1024',
                   'optimizer':'Adam',
                   'learning_rate':'0.0001', 
                   'weight_decay':'0.0', 
                   'dropout_rate':'0.5',
                   'data_augmentation_factor':'1'
                  }
                }
        )  # target2
    sys.argv.append('Musculoskeletal')  # organ
    sys.argv.append('FullBody')  # view
    sys.argv.append('Mixed')  # transformation
    
print(sys.argv)

# Generate results
Attention_Maps_Difference = AttentionMapsDifference(t1=sys.argv[1],                                                         
                                                    t2=sys.argv[2], 
                                                    organ=sys.argv[3],
                                                    view=sys.argv[4], 
                                                    transformation=sys.argv[5])

Attention_Maps_Difference.preprocessing()
Attention_Maps_Difference.process_difference(nb_eids)


# Exit
print('Done.')
sys.exit(0)