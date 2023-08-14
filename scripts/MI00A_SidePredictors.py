import pandas as pd

PATH_UK_BIOBANK = '/n/groups/patel/uk_biobank/'

print('Reading side predictors ...')
usecols = ['eid', '31-0.0', '22001-0.0', '21000-0.0', '21000-1.0', '21000-2.0', '22414-2.0']
data_features_SP = pd.read_csv(PATH_UK_BIOBANK + 'project_52887_41230/ukb41230.csv',usecols=usecols, encoding= 'unicode_escape')

print('Saving side predictors ...')
data_features_SP.to_csv('../data/dataframes/data_features_SP.csv')


