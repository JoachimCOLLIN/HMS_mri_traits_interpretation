# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import sys
from MI_Classes import AttentionMaps


only_guided_gradcam = True
regenerate_saliencies = True
# Options
# Use a small subset of the data VS. run the actual full data pipeline to get accurate results
# /!\ if True, path to save weights will be automatically modified to avoid rewriting them
debug_mode = False

# Default parameters
if len(sys.argv) != 7:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append({
        'METRIC.W.2' : {
            'architecture':'ResNet50',
            'n_fc_layers':1,
            'n_fc_nodes':1024,
            'optimizer':'Adam',
            'learning_rate':0.0001, 
            'weight_decay':0.0, 
            'dropout_rate':0.25,
            'data_augmentation_factor':1
            },
        'METRIC.HEIGHT.2' : {
            'architecture':'ResNet50',
            'n_fc_layers':1,
            'n_fc_nodes':1024,
            'optimizer':'Adam',
            'learning_rate':0.0001, 
            'weight_decay':0.0, 
            'dropout_rate':0.25,
            'data_augmentation_factor':1
            }
        }
    )  # targets
    sys.argv.append('Musculoskeletal')  # organ
    sys.argv.append('FullBody')  # view
    sys.argv.append('Mixed')  # transformation
    sys.argv.append(1000)  # N_samples_attentionmaps   
    sys.argv.append(False)
    
    

print(sys.argv)

# Generate results
Attention_Maps = AttentionMaps(d_targets=sys.argv[1], 
                               organ=sys.argv[2], 
                               view=sys.argv[3], 
                               transformation=sys.argv[4],
                               debug_mode=debug_mode, 
                               N_samples_attentionmaps=sys.argv[5], 
                               only_guided_gradcam=only_guided_gradcam,
                               load_eids_from_version=sys.argv[6], 
                               regenerate_saliencies=regenerate_saliencies)
Attention_Maps.preprocessing()
Attention_Maps.generate_filters()

# Exit
print('Done.')
sys.exit(0)