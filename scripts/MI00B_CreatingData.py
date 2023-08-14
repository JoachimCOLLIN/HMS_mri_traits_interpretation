import pandas as pd
import sys

PATH_DATA = '/n/groups/patel/joachim/PRS_METRICS/'

# Default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    # code type
    sys.argv.append('METRIC.BMI.2.2') 
    
print(sys.argv)
target = sys.argv[1]

data_features_SP = pd.read_csv('../data/dataframes/data_features_SP.csv')
data_features_VAR = pd.read_csv(PATH_DATA + f'data-features_{target}.csv')
col_name = [col for col in data_features_VAR.columns if ('eid' not in col) & ('named' not in col)][0]
print(f'Column name {col_name} will be renamed "target"!')
data_features_VAR = data_features_VAR[['eid', col_name]].rename(columns={col_name:'target'})
data_features = data_features_SP.merge(data_features_VAR, on='eid', how='inner')
data_features.to_csv(f'/n/groups/patel/joachim/data/dataframes/data-features_{target}.csv')
# data_features.to_csv(f'/n/groups/patel/joachim/scripts/data_features_{code_type}_{code}.csv')


    
