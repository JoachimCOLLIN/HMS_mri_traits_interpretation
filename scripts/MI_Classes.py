
# LIBRARIES
# set up backend for ssh -x11 figures
import matplotlib
import cv2

matplotlib.use('Agg')

# read and write
import os
import sys
import glob
import re
import fnmatch
import csv
import shutil
from datetime import datetime

# maths
import numpy as np
import pandas as pd
import math
import random

# miscellaneous
import warnings
import gc
import timeit

# sklearn
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss, roc_auc_score, \
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, average_precision_score
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import KFold, PredefinedSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Statistics
from scipy.stats import pearsonr, ttest_rel, norm
import scipy.stats as stats

# Other tools for ensemble models building (Samuel Diai's InnerCV class)
from hyperopt import fmin, tpe, space_eval, Trials, hp, STATUS_OK
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# CPUs
from multiprocessing import Pool
# GPUs
from GPUtil import GPUtil

# tensorflow
import tensorflow as tf
# keras
import keras
from keras_preprocessing.image import ImageDataGenerator, Iterator
from keras_preprocessing.image.utils import load_img, img_to_array, array_to_img
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, concatenate
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError, AUC, BinaryAccuracy, Precision, Recall, \
    TruePositives, FalsePositives, FalseNegatives, TrueNegatives
from tensorflow_addons.metrics import RSquare, F1Score
from tensorflow.keras import layers

# Plots
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from bioinfokit import visuz
import matplotlib.image as mpimg

# Model's attention
from keract import get_activations, get_gradients_of_activations
from scipy.ndimage.interpolation import zoom

# Survival
from lifelines.utils import concordance_index

# Necessary to define MyCSVLogger
import collections
import csv
import io
import six
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.compat import collections_abc
from tensorflow.keras.backend import eval

class Basics:
    
    """
    Root class herited by most other class. Includes handy helper functions
    """
    
    def __init__(self):
        # seeds for reproducibility
        self.seed = 0
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # other parameters
        self.project = 'PRS'
        self.ipath = '/n/groups/patel/Alan/Aging/Medical_Images/data/'
        self.spath = '../data/'
        self.folds = ['train', 'val', 'test']
        self.n_CV_outer_folds = 10
        self.outer_folds = [str(x) for x in list(range(self.n_CV_outer_folds))]
        self.modes = ['', '_sd', '_str']
        self.id_vars = ['eid', 'outer_fold']
        self.ethnicities_vars_forgot_Other = \
            ['Ethnicity.White', 'Ethnicity.British', 'Ethnicity.Irish', 'Ethnicity.White_Other', 'Ethnicity.Mixed',
             'Ethnicity.White_and_Black_Caribbean', 'Ethnicity.White_and_Black_African', 'Ethnicity.White_and_Asian',
             'Ethnicity.Mixed_Other', 'Ethnicity.Asian', 'Ethnicity.Indian', 'Ethnicity.Pakistani',
             'Ethnicity.Bangladeshi', 'Ethnicity.Asian_Other', 'Ethnicity.Black', 'Ethnicity.Caribbean',
             'Ethnicity.African', 'Ethnicity.Black_Other', 'Ethnicity.Chinese', 'Ethnicity.Other_ethnicity',
             'Ethnicity.Do_not_know', 'Ethnicity.Prefer_not_to_answer', 'Ethnicity.NA']
        self.ethnicities_vars = \
            ['Ethnicity.White', 'Ethnicity.British', 'Ethnicity.Irish', 'Ethnicity.White_Other', 'Ethnicity.Mixed',
             'Ethnicity.White_and_Black_Caribbean', 'Ethnicity.White_and_Black_African', 'Ethnicity.White_and_Asian',
             'Ethnicity.Mixed_Other', 'Ethnicity.Asian', 'Ethnicity.Indian', 'Ethnicity.Pakistani',
             'Ethnicity.Bangladeshi', 'Ethnicity.Asian_Other', 'Ethnicity.Black', 'Ethnicity.Caribbean',
             'Ethnicity.African', 'Ethnicity.Black_Other', 'Ethnicity.Chinese', 'Ethnicity.Other',
             'Ethnicity.Other_ethnicity', 'Ethnicity.Do_not_know', 'Ethnicity.Prefer_not_to_answer', 'Ethnicity.NA']
        self.demographic_vars = ['target', 'Sex'] + self.ethnicities_vars
        self.names_model_parameters = ['target', 'organ', 'view', 'transformation', 'architecture', 'n_fc_layers',
                                       'n_fc_nodes', 'optimizer', 'learning_rate', 'weight_decay', 'dropout_rate',
                                       'data_augmentation_factor']
        self.targets_regression = ['PRS']
        self.targets_binary = ['Sex']
        self.models_types = ['', '_bestmodels']
        self.dict_prediction_types = {'PRS': 'regression', 'Sex': 'binary'}
        self.dict_side_predictors = {'PRS': ['Sex'] + self.ethnicities_vars_forgot_Other,
                                     'Sex': ['target'] + self.ethnicities_vars_forgot_Other}
        self.organs = ['Brain', 'Eyes', 'Arterial', 'Heart', 'Abdomen', 'Musculoskeletal']
        self.left_right_organs_views = ['Eyes_Fundus', 'Eyes_OCT', 'Arterial_Carotids', 'Musculoskeletal_Hips',
                                        'Musculoskeletal_Knees']
        self.dict_organs_to_views = {'Brain': ['MRI'],
                                     'Eyes': ['Fundus', 'OCT'],
                                     'Arterial': ['Carotids'],
                                     'Heart': ['MRI'],
                                     'Abdomen': ['Liver', 'Pancreas'],
                                     'Musculoskeletal': ['Spine', 'Hips', 'Knees', 'FullBody'],
                                     'PhysicalActivity': ['FullWeek']}
        self.dict_organsviews_to_transformations = \
            {'Brain_MRI': ['SagittalRaw', 'SagittalReference', 'CoronalRaw', 'CoronalReference', 'TransverseRaw',
                               'TransverseReference'],
             'Arterial_Carotids': ['Mixed', 'LongAxis', 'CIMT120', 'CIMT150', 'ShortAxis'],
             'Heart_MRI': ['2chambersRaw', '2chambersContrast', '3chambersRaw', '3chambersContrast', '4chambersRaw',
                           '4chambersContrast'],
             'Musculoskeletal_Spine': ['Sagittal', 'Coronal'],
             'Musculoskeletal_FullBody': ['Mixed', 'Figure', 'Skeleton', 'Flesh'],
             'PhysicalActivity_FullWeek': ['GramianAngularField1minDifference', 'GramianAngularField1minSummation',
                                           'MarkovTransitionField1min', 'RecurrencePlots1min']}
        self.dict_organsviews_to_transformations.update(dict.fromkeys(['Eyes_Fundus', 'Eyes_OCT'], ['Raw']))
        self.dict_organsviews_to_transformations.update(
            dict.fromkeys(['Abdomen_Liver', 'Abdomen_Pancreas'], ['Raw', 'Contrast']))
        self.dict_organsviews_to_transformations.update(
            dict.fromkeys(['Musculoskeletal_Hips', 'Musculoskeletal_Knees'], ['MRI']))
        self.organsviews_not_to_augment = []
        self.organs_instances23 = ['Brain', 'Eyes', 'Arterial', 'Heart', 'Abdomen', 'Musculoskeletal',
                                   'PhysicalActivity']
        self.organs_XWAS = \
            ['*', '*instances01', '*instances1.5x', '*instances23', 'Brain', 'BrainCognitive', 'BrainMRI', 'Eyes',
             'EyesFundus', 'EyesOCT', 'Hearing', 'Lungs', 'Arterial', 'ArterialPulseWaveAnalysis', 'ArterialCarotids',
             'Heart', 'HeartECG', 'HeartMRI', 'Abdomen', 'AbdomenLiver', 'AbdomenPancreas', 'Musculoskeletal',
             'MusculoskeletalSpine', 'MusculoskeletalHips', 'MusculoskeletalKnees', 'MusculoskeletalFullBody',
             'MusculoskeletalScalars', 'PhysicalActivity', 'Biochemistry', 'BiochemistryUrine', 'BiochemistryBlood',
             'ImmuneSystem']
        
        gc.enable()  # garbage collector
        warnings.filterwarnings('ignore')
    
    def _version_to_parameters(self, model_name):
        parameters = {}
        parameters_list = model_name.split('_')
        # to change ?
        indices = [i for i, x in enumerate(parameters_list) if 'residual' in x]
        indices.sort()
        for i in range(len(indices)):
            idx = indices[i]
            if idx!=0:
                parameters_list[idx-1] = parameters_list[idx-1] + '_' + parameters_list[idx]
                parameters_list.pop(idx)
                indices = [j-1 for j in indices] 
        for i, parameter in enumerate(self.names_model_parameters):
            parameters[parameter] = parameters_list[i]
        if len(parameters_list) > 12:
            parameters['outer_fold'] = parameters_list[12]
        return parameters
    
    @staticmethod
    def _parameters_to_version(parameters):
        return '_'.join(parameters.values())
    
    @staticmethod
    def convert_string_to_boolean(string):
        if string == 'True':
            boolean = True
        elif string == 'False':
            boolean = False
        else:
            print('ERROR: string must be either \'True\' or \'False\'')
            sys.exit(1)
        return boolean
    
def df_concatenate_column(df, col_concat,l_cols):
    l_df_cols = df.columns.tolist()
    l_cols_to_concat = [col for col in l_cols if col in l_df_cols]
    if len(l_cols_to_concat) > 0:
        concatenation = df[l_cols_to_concat[0]]
        for col in l_cols_to_concat[1:]:
            concatenation = concatenation + df[col]
        df[col_concat] = concatenation
    else:
        df[col_concat] = 0
    return df
            
    
    
class PreprocessingMain(Basics):
    
    """
    This class executes the code for step 01. It preprocesses the main dataframe by:
    - reformating the rows and columns
    - splitting the dataset into folds for the future cross validations
    - imputing key missing data
    - adding a new UKB instance for physical activity data
    - formating the demographics columns (age, sex and ethnicity)
    - reformating the dataframe so that different instances of the same participant are treated as different rows
    - saving the dataframe
    """
    
    def __init__(self, target):
        Basics.__init__(self)
        self.data_raw = None
        self.data_features = None
        self.data_features_eids = None
        self.target = target
    
    def _add_outer_folds(self):
        outer_folds_split = pd.read_csv(self.ipath + 'All_eids.csv')
        outer_folds_split.rename(columns={'fold': 'outer_fold'}, inplace=True)
        outer_folds_split['eid'] = outer_folds_split['eid'].astype('str')
        outer_folds_split['outer_fold'] = outer_folds_split['outer_fold'].astype('str')
        outer_folds_split.set_index('eid', inplace=True)
        self.data_raw = self.data_raw.join(outer_folds_split)
    
    def _compute_sex(self):
        # Use genetic sex when available
        self.data_raw['Sex_genetic'][self.data_raw['Sex_genetic'].isna()] = \
            self.data_raw['Sex'][self.data_raw['Sex_genetic'].isna()]
        self.data_raw.drop(['Sex'], axis=1, inplace=True)
        self.data_raw.rename(columns={'Sex_genetic': 'Sex'}, inplace=True)
        self.data_raw.dropna(subset=['Sex'], inplace=True)
    
    def _compute_target(self):
        # print('data-features_computed before removing PRS NAN:', self.data_raw.shape[0])
        self.data_raw.dropna(subset=['target'], inplace=True)
        # print('data-features_computed after removing PRS NAN:', self.data_raw.shape[0])
    
    def _encode_ethnicity(self):
        # Fill NAs for ethnicity on instance 0 if available in other instances
        eids_missing_ethnicity = self.data_raw['eid'][self.data_raw['Ethnicity'].isna()]
        for eid in eids_missing_ethnicity:
            sample = self.data_raw.loc[eid, :]
            if not math.isnan(sample['Ethnicity_1']):
                self.data_raw.loc[eid, 'Ethnicity'] = self.data_raw.loc[eid, 'Ethnicity_1']
            elif not math.isnan(sample['Ethnicity_2']):
                self.data_raw.loc[eid, 'Ethnicity'] = self.data_raw.loc[eid, 'Ethnicity_2']
        self.data_raw.drop(['Ethnicity_1', 'Ethnicity_2'], axis=1, inplace=True)
        
        # One hot encode ethnicity
        dict_ethnicity_codes = {'1': 'Ethnicity.White', '1001': 'Ethnicity.British', '1002': 'Ethnicity.Irish',
                                '1003': 'Ethnicity.White_Other',
                                '2': 'Ethnicity.Mixed', '2001': 'Ethnicity.White_and_Black_Caribbean',
                                '2002': 'Ethnicity.White_and_Black_African',
                                '2003': 'Ethnicity.White_and_Asian', '2004': 'Ethnicity.Mixed_Other',
                                '3': 'Ethnicity.Asian', '3001': 'Ethnicity.Indian', '3002': 'Ethnicity.Pakistani',
                                '3003': 'Ethnicity.Bangladeshi', '3004': 'Ethnicity.Asian_Other',
                                '4': 'Ethnicity.Black', '4001': 'Ethnicity.Caribbean', '4002': 'Ethnicity.African',
                                '4003': 'Ethnicity.Black_Other',
                                '5': 'Ethnicity.Chinese',
                                '6': 'Ethnicity.Other_ethnicity',
                                '-1': 'Ethnicity.Do_not_know',
                                '-3': 'Ethnicity.Prefer_not_to_answer',
                                '-5': 'Ethnicity.NA'}
        self.data_raw['Ethnicity'] = self.data_raw['Ethnicity'].fillna(-5).astype(int).astype(str)
        ethnicities = pd.get_dummies(self.data_raw['Ethnicity'])
        self.data_raw.drop(['Ethnicity'], axis=1, inplace=True)
        ethnicities.rename(columns=dict_ethnicity_codes, inplace=True)
        
        
        
        
        ethnicities = df_concatenate_column(ethnicities, 'Ethnicity.White', ['Ethnicity.White', 'Ethnicity.British', 'Ethnicity.Irish', 'Ethnicity.White_Other'])
        ethnicities = df_concatenate_column(ethnicities, 'Ethnicity.Mixed', ['Ethnicity.Mixed', 'Ethnicity.White_and_Black_Caribbean', 'Ethnicity.White_and_Black_African', 'Ethnicity.White_and_Asian', 'Ethnicity.Mixed_Other'])
        ethnicities = df_concatenate_column(ethnicities, 'Ethnicity.Asian', ['Ethnicity.Asian', 'Ethnicity.Indian', 'Ethnicity.Pakistani', 'Ethnicity.Bangladeshi', 'Ethnicity.Asian_Other'])
        ethnicities = df_concatenate_column(ethnicities, 'Ethnicity.Black', ['Ethnicity.Black', 'Ethnicity.Caribbean', 'Ethnicity.African', 'Ethnicity.Black_Other'])
        ethnicities = df_concatenate_column(ethnicities, 'Ethnicity.Other', ['Ethnicity.Other_ethnicity', 'Ethnicity.Do_not_know', 'Ethnicity.Prefer_not_to_answer', 'Ethnicity.NA'])
        
        # ethnicities['Ethnicity.White'] = ethnicities['Ethnicity.White'] + ethnicities['Ethnicity.British'] + \
        #                                  ethnicities['Ethnicity.Irish'] + ethnicities['Ethnicity.White_Other']
        # ethnicities['Ethnicity.Mixed'] = ethnicities['Ethnicity.Mixed'] + \
        #                                  ethnicities['Ethnicity.White_and_Black_Caribbean'] + \
        #                                  ethnicities['Ethnicity.White_and_Black_African'] + \
        #                                  ethnicities['Ethnicity.White_and_Asian'] + \
        #                                  ethnicities['Ethnicity.Mixed_Other']
        # ethnicities['Ethnicity.Asian'] = ethnicities['Ethnicity.Asian'] + ethnicities['Ethnicity.Indian'] + \
        #                                  ethnicities['Ethnicity.Pakistani'] + ethnicities['Ethnicity.Bangladeshi'] + \
        #                                  ethnicities['Ethnicity.Asian_Other']
        # ethnicities['Ethnicity.Black'] = ethnicities['Ethnicity.Black'] + ethnicities['Ethnicity.Caribbean'] + \
        #                                  ethnicities['Ethnicity.African'] + ethnicities['Ethnicity.Black_Other']
        # ethnicities['Ethnicity.Other'] = ethnicities['Ethnicity.Other_ethnicity'] + \
        #                                  ethnicities['Ethnicity.Do_not_know'] + \
        #                                  ethnicities['Ethnicity.Prefer_not_to_answer'] + \
        #                                  ethnicities['Ethnicity.NA']
        self.data_raw = self.data_raw.join(ethnicities)
    
    def generate_data(self):
        dict_UKB_fields_to_names = {'31-0.0': 'Sex', '22001-0.0': 'Sex_genetic', '21000-0.0': 'Ethnicity',
                                    '21000-1.0': 'Ethnicity_1', '21000-2.0': 'Ethnicity_2',
                                    '22414-2.0': 'Abdominal_images_quality'}
        self.data_raw = pd.read_csv(self.spath + f'dataframes/data-features_{self.target}.csv',
                                    usecols=['eid', '31-0.0', '22001-0.0', '21000-0.0', '21000-1.0', '21000-2.0',
                                             'target', '22414-2.0'])
        # Formatting
        self.data_raw.rename(columns=dict_UKB_fields_to_names, inplace=True)
        self.data_raw['eid'] = self.data_raw['eid'].astype(str)
        self.data_raw.set_index('eid', drop=False, inplace=True)
        self.data_raw.index.name = 'column_names'
        self._add_outer_folds()
        self._compute_sex()
        self._encode_ethnicity()
        self._compute_target()
        
        df = self.data_raw.copy()
        # filling missing columns with 0
        l_demegraphic_vars_not_in = [col_name for col_name in self.demographic_vars if col_name not in df.columns.tolist()]
        df[l_demegraphic_vars_not_in] = 0
        df = df[self.id_vars + self.demographic_vars + ['Abdominal_images_quality']]
        self.data_features = df
        
        
        # Save age as a float32 instead of float64
        self.data_features['target'] = np.float32(self.data_features['target'])
        
        # Shuffle the rows before saving the dataframe
        self.data_features = self.data_features.sample(frac=1)
        
    
    def save_data(self):
        print(f'Saving restults at  /dataframes/data-features_computed_{self.target}.csv ...')
        self.data_features.to_csv(self.spath + f'/dataframes/data-features_computed_{self.target}.csv', index=False)

# Helper functions
def pearsoncor(x, y):
    r, _ = pearsonr(x, y)
    return r

def df_ids_selection_from_list(df, l_ids):
    #select ids of df that are in list 
    df_new = df[df.index.isin(l_ids)]
    return df_new
class Metrics(Basics):
    
    """
    Helper class defining dictionaries of metrics and custom metrics
    """
    
    def __init__(self):
        # Parameters
        Basics.__init__(self)
        self.metrics_displayed_in_int = ['True-Positives', 'True-Negatives', 'False-Positives', 'False-Negatives']
        self.metrics_needing_classpred = ['F1-Score', 'Binary-Accuracy', 'Precision', 'Recall']
        self.dict_metrics_names_K = {'regression': ['RMSE'],  # For now, R-Square is buggy. Try again in a few months.
                                     'binary': ['ROC-AUC', 'PR-AUC', 'F1-Score', 'Binary-Accuracy', 'Precision',
                                                'Recall', 'True-Positives', 'False-Positives', 'False-Negatives',
                                                'True-Negatives'],
                                     'multiclass': ['Categorical-Accuracy']}
        self.dict_metrics_names = {'regression': ['MAE', 'RMSE', 'R-Squared', 'Pearson-Correlation'],
                                   'binary': ['ROC-AUC', 'F1-Score', 'PR-AUC', 'Binary-Accuracy', 'Sensitivity',
                                              'Specificity', 'Precision', 'Recall', 'True-Positives', 'False-Positives',
                                              'False-Negatives', 'True-Negatives'],
                                   'multiclass': ['Categorical-Accuracy']}
        self.dict_losses_names = {'regression': 'MSE', 'binary': 'Binary-Crossentropy',
                                  'multiclass': 'categorical_crossentropy'}
        self.dict_main_metrics_names_K = {'PRS': 'RMSE', 'Sex': 'PR-AUC', 'imbalanced_binary_placeholder': 'PR-AUC'}
        self.dict_main_metrics_names = {'PRS': 'R-Squared', 'Sex': 'ROC-AUC',
                                        'imbalanced_binary_placeholder': 'PR-AUC'}
        self.main_metrics_modes = {'loss': 'min', 'R-Squared': 'max', 'Pearson-Correlation': 'max', 'RMSE': 'min',
                                   'MAE': 'min', 'ROC-AUC': 'max', 'PR-AUC': 'max', 'F1-Score': 'max', 'C-Index': 'max',
                                   'C-Index-difference': 'max'}
        
        self.n_bootstrap_iterations = 1000
        
        def rmse(y_true, y_pred):
            return math.sqrt(mean_squared_error(y_true, y_pred))
        
        def sensitivity_score(y, pred):
            _, _, fn, tp = confusion_matrix(y, pred.round()).ravel()
            return tp / (tp + fn)
        
        def specificity_score(y, pred):
            tn, fp, _, _ = confusion_matrix(y, pred.round()).ravel()
            return tn / (tn + fp)
        
        def true_positives_score(y, pred):
            _, _, _, tp = confusion_matrix(y, pred.round()).ravel()
            return tp
        
        def false_positives_score(y, pred):
            _, fp, _, _ = confusion_matrix(y, pred.round()).ravel()
            return fp
        
        def false_negatives_score(y, pred):
            _, _, fn, _ = confusion_matrix(y, pred.round()).ravel()
            return fn
        
        def true_negatives_score(y, pred):
            tn, _, _, _ = confusion_matrix(y, pred.round()).ravel()
            return tn
        
        self.dict_metrics_sklearn = {'mean_squared_error': mean_squared_error,
                                     'MAE': mean_absolute_error,
                                     'RMSE': rmse,
                                     'Pearson-Correlation': pearsoncor,
                                     'R-Squared': r2_score,
                                     'Binary-Crossentropy': log_loss,
                                     'ROC-AUC': roc_auc_score,
                                     'F1-Score': f1_score,
                                     'PR-AUC': average_precision_score,
                                     'Binary-Accuracy': accuracy_score,
                                     'Sensitivity': sensitivity_score,
                                     'Specificity': specificity_score,
                                     'Precision': precision_score,
                                     'Recall': recall_score,
                                     'True-Positives': true_positives_score,
                                     'False-Positives': false_positives_score,
                                     'False-Negatives': false_negatives_score,
                                     'True-Negatives': true_negatives_score}
    
    def _bootstrap(self, data, function):
        results = []
        for i in range(self.n_bootstrap_iterations):
            data_i = resample(data, replace=True, n_samples=len(data.index))
            results.append(function(data_i['y'], data_i['pred']))
        return np.mean(results), np.std(results)
    
    
class PreprocessingFolds(Metrics):
    
    """
    Splits the data into training, validation and testing sets for all CV folds
    """
    
    def __init__(self, target, organ, regenerate_data):
        Metrics.__init__(self)
        self.organ = organ
        self.target = target
        self.list_ids_per_view_transformation = None
        
        # Check if these folds have already been generated
        if not regenerate_data:
            if len(glob.glob(self.spath + 'dataframes/data-features_' + target + '_' + organ + '_*.csv')) > 0:
                print("Error: The files already exist! Either change regenerate_data to True or delete the previous"
                      " version.")
                sys.exit(1)
        
        self.side_predictors = self.dict_side_predictors[self.project]
        self.variables_to_normalize = self.side_predictors
        if self.project in self.targets_regression:
            self.variables_to_normalize.append('target')
        self.dict_image_quality_col = {'Liver': 'Abdominal_images_quality'}
        self.dict_image_quality_col.update(
            dict.fromkeys(['Brain', 'Eyes', 'Arterial', 'Heart', 'Abdomen', 'Musculoskeletal', 'PhysicalActivity'],
                          None))
        self.image_quality_col = self.dict_image_quality_col[organ]
        self.views = self.dict_organs_to_views[organ]
        self.list_ids = None
        self.list_ids_per_view = {}
        self.data = None
        self.EIDS = None
        self.EIDS_per_view = {'train': {}, 'val': {}, 'test': {}}
        self.data_fold = None
    
    def _get_list_ids(self):
        self.list_ids_per_view_transformation = {}
        list_ids = []
        # if different views are available, take the union of the ids
        for view in self.views:
            self.list_ids_per_view_transformation[view] = {}
            for transformation in self.dict_organsviews_to_transformations[self.organ + '_' + view]:
                list_ids_transformation = []
                path = self.ipath + '../images/' + self.organ + '/' + view + '/' + transformation + '/'
                # for paired organs, take the unions of the ids available on the right and the left sides
                if self.organ + '_' + view in self.left_right_organs_views:
                    for side in ['right', 'left']:
                        list_ids_transformation += os.listdir(path + side + '/')
                    list_ids_transformation = np.unique(list_ids_transformation).tolist()
                else:
                    list_ids_transformation += os.listdir(path)
                #don't care about instance
                self.list_ids_per_view_transformation[view][transformation] = \
                    [im.replace('.jpg', '').split('_')[0] for im in list_ids_transformation]
                list_ids += self.list_ids_per_view_transformation[view][transformation]
        self.list_ids = np.unique(list_ids).tolist()
        self.list_ids.sort()
        print('Number of unique eids (all instances) for this organ (all views and transformations):', len(self.list_ids))
    
    def _filter_and_format_data(self):
        """
        Clean the data before it can be split between the rows
        """
        cols_data = self.id_vars + self.demographic_vars
        if self.image_quality_col is not None:
            cols_data.append(self.dict_image_quality_col[self.organ])
        data = pd.read_csv(self.spath + f'dataframes/data-features_computed_{self.target}.csv', usecols=cols_data)
        data.rename(columns={self.dict_image_quality_col[self.organ]: 'Data_quality'}, inplace=True)
        for col_name in self.id_vars:
            data[col_name] = data[col_name].astype(str)
        data.set_index('eid', drop=False, inplace=True)
        if self.image_quality_col is not None:
            data = data[data['Data_quality'] != np.nan]
            data.drop('Data_quality', axis=1, inplace=True)
        # get rid of samples with NAs
        print('Data-features_computed size before removing all NaNs:', data.shape[0])
        data.dropna(inplace=True)
        print('Data-features_computed size after removing all NaNs::', data.shape[0])
        # list the samples' ids for which images are available
        # data = data.loc[self.list_ids]
        data = df_ids_selection_from_list(data, self.list_ids)
        list_images_not_in_data = [eid for eid in self.list_ids if eid not in data.index.values]
        print('Number of eids not in data-features_computed:', len(list_images_not_in_data))
        self.data = data
        print('New data-features size:', self.data.shape[0])
    
    def _split_data(self):
        # Generate the data for each outer_fold
        for i, outer_fold in enumerate(self.outer_folds):
            of_val = outer_fold
            of_test = str((int(outer_fold) + 1) % len(self.outer_folds))
            DATA = {
                'train': self.data[~self.data['outer_fold'].isin([of_val, of_test])],
                'val': self.data[self.data['outer_fold'] == of_val],
                'test': self.data[self.data['outer_fold'] == of_test]
            }
            
            # Generate the data for the different views and transformations
            for view in self.views:
                for transformation in self.dict_organsviews_to_transformations[self.organ + '_' + view]:
                    print('Splitting data for view ' + view + ', and transformation ' + transformation)
                    DF = {}
                    for fold in self.folds:
                        idx = DATA[fold]['eid'].isin(self.list_ids_per_view_transformation[view][transformation]).values
                        DF[fold] = DATA[fold].iloc[idx, :]
                    
                    # compute values for scaling of variables
                    normalizing_values = {}
                    for var in self.variables_to_normalize:
                        var_mean = DF['train'][var].mean()
                        if len(DF['train'][var].unique()) < 2:
                            print('Variable ' + var + ' has a single value in fold ' + outer_fold +
                                  '. Using 1 as std for normalization.')
                            var_std = 1
                        else:
                            var_std = DF['train'][var].std()
                        normalizing_values[var] = {'mean': var_mean, 'std': var_std}

                    # normalize the variables
                    for fold in self.folds:
                        for var in self.variables_to_normalize:
                            DF[fold][var + '_raw'] = DF[fold][var]
                            DF[fold][var] = (DF[fold][var] - normalizing_values[var]['mean']) \
                                             / normalizing_values[var]['std']

                        # report issue if NAs were detected (most likely comes from a sample whose id did not match)
                        n_mismatching_samples = DF[fold].isna().sum().max()
                        if n_mismatching_samples > 0:
                            print(DF[fold][DF[fold].isna().any(axis=1)])
                            print('/!\\ WARNING! ' + str(n_mismatching_samples) + ' ' + fold + ' images ids out of ' +
                                  str(len(DF[fold].index)) + ' did not match the dataframe!')
                        
                        # save the data
                        DF[fold].to_csv(self.spath + 'dataframes/data-features_' + self.target + '_' + self.organ + '_' + view + '_' +
                                         transformation + '_' + fold + '_' + outer_fold + '.csv',
                                         index=False)
                        print('For outer_fold ' + outer_fold + ', the ' + fold + ' fold has a sample size of ' +
                              str(len(DF[fold].index)))
    
    def generate_folds(self):
        self._get_list_ids()
        self._filter_and_format_data()
        self._split_data()

        
class MyImageDataGenerator(Basics, Sequence, ImageDataGenerator):
    
    """
    Helper class: custom data generator for images.
    It handles several custom features such as:
    - provides batches of not only images, but also the scalar data (e.g demographics) that correspond to it
    - it performs random shuffling while making sure that no leftover data (the remainder of the modulo batch size)
        is being unused
    - it can handle paired data for paired organs (e.g left/right eyes)
    """
    
    def __init__(self, organ=None, view=None, data_features=None, n_samples_per_subepoch=None,
                 batch_size=None, training_mode=None, side_predictors=None, dir_images=None, images_width=None,
                 images_height=None, data_augmentation=False, data_augmentation_factor=None, seed=None):
        # Parameters
        Basics.__init__(self)
        if self.project in self.targets_regression:
            # normalized data
            self.labels = data_features['target']
        # ?? error ??
        else:
            self.labels = data_features['target' + '_raw']
        self.organ = organ
        self.view = view
        self.training_mode = training_mode
        self.data_features = data_features
        self.list_ids = data_features.index.values
        self.batch_size = batch_size
        # for paired organs, take twice fewer ids (two images for each id), and add organ_side as side predictor
        if organ + '_' + view in self.left_right_organs_views:
            self.data_features['organ_side'] = np.nan
            self.n_ids_batch = batch_size // 2
        else:
            self.n_ids_batch = batch_size
        if self.training_mode & (n_samples_per_subepoch is not None):  # during training, 1 epoch = number of samples
            self.steps = math.ceil(n_samples_per_subepoch / batch_size)
        else:  # during prediction and other tasks, an epoch is defined as all the samples being seen once and only once
            self.steps = math.ceil(len(self.list_ids) / self.n_ids_batch)
        # learning_rate_patience
        if n_samples_per_subepoch is not None:
            self.n_subepochs_per_epoch = math.ceil(len(self.data_features.index) / n_samples_per_subepoch)
        # initiate the indices and shuffle the ids
        self.shuffle = training_mode  # Only shuffle if the model is being trained. Otherwise no need.
        self.indices = np.arange(len(self.list_ids))
        self.idx_end = 0  # Keep track of last indice to permute indices accordingly at the end of epoch.
        if self.shuffle:
            np.random.shuffle(self.indices)
        # Input for side NN and CNN
        self.side_predictors = side_predictors
        self.dir_images = dir_images
        self.images_width = images_width
        self.images_height = images_height
        # Data augmentation
        self.data_augmentation = data_augmentation
        self.data_augmentation_factor = data_augmentation_factor
        self.seed = seed
        # Parameters for data augmentation: (rotation range, width shift range, height shift range, zoom range)
        self.augmentation_parameters = \
            pd.DataFrame(index=['Brain_MRI', 'Eyes_Fundus', 'Eyes_OCT', 'Arterial_Carotids', 'Heart_MRI',
                                'Abdomen_Liver', 'Abdomen_Pancreas', 'Musculoskeletal_Spine', 'Musculoskeletal_Hips',
                                'Musculoskeletal_Knees', 'Musculoskeletal_FullBody', 'PhysicalActivity_FullWeek',
                                'PhysicalActivity_Walking'],
                         columns=['rotation', 'width_shift', 'height_shift', 'zoom'])
        self.augmentation_parameters.loc['Brain_MRI', :] = [10, 0.05, 0.1, 0.0]
        self.augmentation_parameters.loc['Eyes_Fundus', :] = [20, 0.02, 0.02, 0]
        self.augmentation_parameters.loc['Eyes_OCT', :] = [30, 0.1, 0.2, 0]
        self.augmentation_parameters.loc[['Arterial_Carotids'], :] = [0, 0.2, 0.0, 0.0]
        self.augmentation_parameters.loc[['Heart_MRI', 'Abdomen_Liver', 'Abdomen_Pancreas',
                                          'Musculoskeletal_Spine'], :] = [10, 0.1, 0.1, 0.0]
        self.augmentation_parameters.loc[['Musculoskeletal_Hips', 'Musculoskeletal_Knees'], :] = [10, 0.1, 0.1, 0.1]
        self.augmentation_parameters.loc[['Musculoskeletal_FullBody'], :] = [10, 0.05, 0.02, 0.0]
        self.augmentation_parameters.loc[['PhysicalActivity_FullWeek'], :] = [0, 0, 0, 0.0]
        organ_view = organ + '_' + view
        ImageDataGenerator.__init__(self, rescale=1. / 255.,
                                    rotation_range=self.augmentation_parameters.loc[organ_view, 'rotation'],
                                    width_shift_range=self.augmentation_parameters.loc[organ_view, 'width_shift'],
                                    height_shift_range=self.augmentation_parameters.loc[organ_view, 'height_shift'],
                                    zoom_range=self.augmentation_parameters.loc[organ_view, 'zoom'])
    
    def __len__(self):
        return self.steps
    
    def on_epoch_end(self):
        _ = gc.collect()
        self.indices = np.concatenate([self.indices[self.idx_end:], self.indices[:self.idx_end]])
    
    def _generate_image(self, path_image):
        try:
            img = load_img(path_image[:-4] + '_2.jpg', target_size=(self.images_width, self.images_height), color_mode='rgb')
        except:
            img = load_img(path_image[:-4] + '_3.jpg', target_size=(self.images_width, self.images_height), color_mode='rgb')
        Xi = img_to_array(img)
        if hasattr(img, 'close'):
            img.close()
        if self.data_augmentation:
            params = self.get_random_transform(Xi.shape)
            Xi = self.apply_transform(Xi, params)
        Xi = self.standardize(Xi)
        return Xi
    
    def _data_generation(self, list_ids_batch):
        # initialize empty matrices
        n_samples_batch = min(len(list_ids_batch), self.batch_size)
        X = np.empty((n_samples_batch, self.images_width, self.images_height, 3)) * np.nan
        x = np.empty((n_samples_batch, len(self.side_predictors))) * np.nan
        y = np.empty((n_samples_batch, 1)) * np.nan
        # fill the matrices sample by sample
        for i, ID in enumerate(list_ids_batch):
            y[i] = self.labels[ID]
            x[i] = self.data_features.loc[ID, self.side_predictors]
            #only one left or right
            if self.organ + '_' + self.view in self.left_right_organs_views:
                if i % 2 == 0:
                    path = self.dir_images + 'right/'
                    x[i][-1] = 0
                else:
                    path = self.dir_images + 'left/'
                    x[i][-1] = 1
                if not os.path.exists(path + ID + '.jpg'):
                    path = path.replace('/right/', '/left/') if i % 2 == 0 else path.replace('/left/', '/right/')
                    x[i][-1] = 1 - x[i][-1]
            else:
                path = self.dir_images
            X[i, :, :, :] = self._generate_image(path_image=path + str(ID) + '.jpg')
        return [X, x], y
    
    def __getitem__(self, index):
        # Select the indices
        idx_start = (index * self.n_ids_batch) % len(self.list_ids)
        idx_end = (((index + 1) * self.n_ids_batch) - 1) % len(self.list_ids) + 1
        if idx_start > idx_end:
            # If this happens outside of training, that is a mistake
            if not self.training_mode:
                print('\nERROR: Outside of training, every sample should only be predicted once!')
                sys.exit(1)
            # Select part of the indices from the end of the epoch
            indices = self.indices[idx_start:]
            # Generate a new set of indices
            # print('\nThe end of the data was reached within this batch, looping.')
            if self.shuffle:
                np.random.shuffle(self.indices)
            # Complete the batch with samples from the new indices
            indices = np.concatenate([indices, self.indices[:idx_end]])
        else:
            indices = self.indices[idx_start: idx_end]
            if idx_end == len(self.list_ids) & self.shuffle:
                # print('\nThe end of the data was reached. Shuffling for the next epoch.')
                np.random.shuffle(self.indices)
        # Keep track of last indice for end of subepoch
        self.idx_end = idx_end
        # Select the corresponding ids
        list_ids_batch = [self.list_ids[i] for i in indices]
        # For paired organs, two images (left, right eyes) are selected for each id.
        if self.organ + '_' + self.view in self.left_right_organs_views:
            list_ids_batch = [ID for ID in list_ids_batch for _ in ('right', 'left')]
        return self._data_generation(list_ids_batch)

    
class MyCSVLogger(Callback):
    
    """
    Custom CSV Logger callback class for Keras training: append to existing file if can be found. Allows to keep track
    of training over several jobs.
    """
    
    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.csv_file = None
        if six.PY2:
            self.file_flags = 'b'
            self._open_args = {}
        else:
            self.file_flags = ''
            self._open_args = {'newline': '\n'}
        Callback.__init__(self)
    
    def on_train_begin(self, logs=None):
        if self.append:
            if file_io.file_exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename, mode + self.file_flags, **self._open_args)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, collections_abc.Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k
        
        if self.keys is None:
            self.keys = sorted(logs.keys())
        
        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])
        
        if not self.writer:
            
            class CustomDialect(csv.excel):
                delimiter = self.sep
            
            fieldnames = ['epoch', 'learning_rate'] + self.keys
            if six.PY2:
                fieldnames = [unicode(x) for x in fieldnames]
            
            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=fieldnames,
                dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()
        
        row_dict = collections.OrderedDict({'epoch': epoch, 'learning_rate': eval(self.model.optimizer.lr)})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()
    
    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None

        
class MyModelCheckpoint(ModelCheckpoint):
    
    """
    Custom checkpoint callback class for Keras training. Handles a baseline performance.
    """
    
    def __init__(self, filepath, monitor='val_loss', baseline=-np.Inf, verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto', save_freq='epoch'):
        # Parameters
        ModelCheckpoint.__init__(self, filepath, monitor=monitor, verbose=verbose, save_best_only=save_best_only,
                                 save_weights_only=save_weights_only, mode=mode, save_freq=save_freq)
        if mode == 'min':
            self.monitor_op = np.less
            self.best = baseline
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = baseline
        else:
            print('Error. mode for metric must be either min or max')
            sys.exit(1)

            
            
def local_to_alans(path, taking_alans_weights):
    
    if taking_alans_weights:
        replace_dict = {'../data/': '/n/groups/patel/Alan/Aging/Medical_Images/data/',
                        '/dataframes':'',
                        '/weights':'', 
                        '_PRS_':'_Age'}
        new_path = path
        for old, new in replace_dict.items():
            new_path = new_path.replace(old, new)
    else:
        new_path=path
    
    return new_path
            
    
    
class DeepLearning(Metrics):
    
    """
    Core helper class to train models. Used to:
    - build the data generators
    - generate the CNN architectures
    - load the weights
    """
    
    def __init__(self, target=None, organ=None, view=None, transformation=None, architecture=None, n_fc_layers=None,
                 n_fc_nodes=None, optimizer=None, learning_rate=None, weight_decay=None, dropout_rate=None,
                 data_augmentation_factor=None, debug_mode=False):
        # Initialization
        Metrics.__init__(self)
        tf.random.set_seed(self.seed)
        
        # Model's version
        self.target = target
        self.organ = organ
        self.view = view
        self.transformation = transformation
        self.architecture = architecture
        self.n_fc_layers = int(n_fc_layers)
        self.n_fc_nodes = int(n_fc_nodes)
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout_rate = float(dropout_rate)
        self.data_augmentation_factor = float(data_augmentation_factor)
        self.outer_fold = None
        self.version =  self.target + '_' + organ + '_' + view + '_' + transformation + '_' + architecture + '_' + \
                       n_fc_layers + '_' + n_fc_nodes + '_' + optimizer + '_' + learning_rate + '_' + weight_decay + \
                       '_' + dropout_rate + '_' + data_augmentation_factor
        
        # NNet's architecture and weights
        self.side_predictors = self.dict_side_predictors[self.project]
        if self.organ + '_' + self.view in self.left_right_organs_views:
            self.side_predictors.append('organ_side')
        self.dict_final_activations = {'regression': 'linear', 'binary': 'sigmoid', 'multiclass': 'softmax',
                                       'saliency': 'linear'}
        self.path_load_weights = None
        self.keras_weights = None
        
        # Generators
        self.debug_mode = debug_mode
        self.debug_fraction = 0.005
        self.DATA_FEATURES = {}
        self.mode = None
        self.n_cpus = len(os.sched_getaffinity(0))
        self.dir_images = self.ipath + '../images/' + organ + '/' + view + '/' + transformation + '/'
        
        # define dictionary to fit the architecture's input size to the images sizes (take min (height, width))
        self.dict_organ_view_transformation_to_image_size = {
            'Eyes_Fundus_Raw': (316, 316),  # initial size (1388, 1388)
            'Eyes_OCT_Raw': (312, 320),  # initial size (500, 512)
            'Musculoskeletal_Spine_Sagittal': (466, 211),  # initial size (1513, 684)
            'Musculoskeletal_Spine_Coronal': (315, 313),  # initial size (724, 720)
            'Musculoskeletal_Hips_MRI': (329, 303),  # initial size (626, 680)
            'Musculoskeletal_Knees_MRI': (347, 286)  # initial size (851, 700)
        }
        self.dict_organ_view_transformation_to_image_size.update(
            dict.fromkeys(['Brain_MRI_SagittalRaw', 'Brain_MRI_SagittalReference', 'Brain_MRI_CoronalRaw',
                           'Brain_MRI_CoronalReference', 'Brain_MRI_TransverseRaw', 'Brain_MRI_TransverseReference'],
                          (316, 316)))  # initial size (88, 88)
        self.dict_organ_view_transformation_to_image_size.update(
            dict.fromkeys(['Arterial_Carotids_Mixed', 'Arterial_Carotids_LongAxis', 'Arterial_Carotids_CIMT120',
                           'Arterial_Carotids_CIMT150', 'Arterial_Carotids_ShortAxis'],
                          (337, 291)))  # initial size (505, 436)
        self.dict_organ_view_transformation_to_image_size.update(
            dict.fromkeys(['Heart_MRI_2chambersRaw', 'Heart_MRI_2chambersContrast', 'Heart_MRI_3chambersRaw',
                           'Heart_MRI_3chambersContrast', 'Heart_MRI_4chambersRaw', 'Heart_MRI_4chambersContrast'],
                          (316, 316)))  # initial size (200, 200)
        self.dict_organ_view_transformation_to_image_size.update(
            dict.fromkeys(['Abdomen_Liver_Raw', 'Abdomen_Liver_Contrast'], (288, 364)))  # initial size (288, 364)
        self.dict_organ_view_transformation_to_image_size.update(
            dict.fromkeys(['Abdomen_Pancreas_Raw', 'Abdomen_Pancreas_Contrast'], (288, 350)))  # initial size (288, 350)
        self.dict_organ_view_transformation_to_image_size.update(
            dict.fromkeys(['Musculoskeletal_FullBody_Figure', 'Musculoskeletal_FullBody_Skeleton',
                           'Musculoskeletal_FullBody_Flesh', 'Musculoskeletal_FullBody_Mixed'],
                          (541, 181)))  # initial size (811, 272)
        self.dict_organ_view_transformation_to_image_size.update(
            dict.fromkeys(['PhysicalActivity_FullWeek_GramianAngularField1minDifference',
                           'PhysicalActivity_FullWeek_GramianAngularField1minSummation',
                           'PhysicalActivity_FullWeek_MarkovTransitionField1min',
                           'PhysicalActivity_FullWeek_RecurrencePlots1min'],
                          (316, 316)))  # initial size (316, 316)
        self.dict_architecture_to_image_size = {'MobileNet': (224, 224), 'MobileNetV2': (224, 224),
                                                'NASNetMobile': (224, 224), 'NASNetLarge': (331, 331)}
        if self.architecture in ['MobileNet', 'MobileNetV2', 'NASNetMobile', 'NASNetLarge']:
            self.image_width, self.image_height = self.dict_architecture_to_image_size[architecture]
        else:
            self.image_width, self.image_height = \
                self.dict_organ_view_transformation_to_image_size[organ + '_' + view + '_' + transformation]
        
        # define dictionary of batch sizes to fit as many samples as the model's architecture allows
        self.dict_batch_sizes = {
            # Default, applies to all images with resized input ~100,000 pixels
            'Default': {'VGG16': 32, 'VGG19': 32, 'DenseNet121': 16, 'DenseNet169': 16, 'DenseNet201': 16,
                        'Xception': 32, 'InceptionV3': 32, 'InceptionResNetV2': 8, 'ResNet50': 32, 'ResNet101': 16,
                        'ResNet152': 16, 'ResNet50V2': 32, 'ResNet101V2': 16, 'ResNet152V2': 16, 'ResNeXt50': 4,
                        'ResNeXt101': 8, 'EfficientNetB7': 4,
                        'MobileNet': 128, 'MobileNetV2': 64, 'NASNetMobile': 64, 'NASNetLarge': 4}}
        # Define batch size
        if organ + '_' + view in self.dict_batch_sizes.keys():
            randoself.batch_size = self.dict_batch_sizes[organ + '_' + view][architecture]
        else:
            self.batch_size = self.dict_batch_sizes['Default'][architecture]
        # double the batch size for the teslaM40 cores that have bigger memory
        if len(GPUtil.getGPUs()) > 0:  # make sure GPUs are available (not truesometimes for debugging)
            if GPUtil.getGPUs()[0].memoryTotal > 20000:
                self.batch_size *= 2
        # Define number of ids per batch (twice fewer for paired organs, because left and right samples)
        self.n_ids_batch = self.batch_size
        if organ + '_' + view in self.left_right_organs_views:
            self.n_ids_batch //= 2
        
        # Define number of samples per subepoch
        if debug_mode:
            self.n_samples_per_subepoch = self.batch_size * 4
        else:
            self.n_samples_per_subepoch = 32768
        if organ + '_' + view in self.left_right_organs_views:
            self.n_samples_per_subepoch //= 2
        
        # dict to decide which field is used to generate the ids when several targets share the same ids
        self.dict_target_to_ids = dict.fromkeys(['PRS', 'Sex'], 'PRS')
        
        # Note: R-Squared and F1-Score are not available, because their batch based values are misleading.
        # For some reason, Sensitivity and Specificity are not available either. Might implement later.
        self.dict_losses_K = {'MSE': MeanSquaredError(name='MSE'),
                              'Binary-Crossentropy': BinaryCrossentropy(name='Binary-Crossentropy')}
        self.dict_metrics_K = {'R-Squared': RSquare(name='R-Squared'), #, y_shape=(1,)
                               'MAE': MeanAbsoluteError(name='MAE'),
                               'RMSE': RootMeanSquaredError(name='RMSE'),
                               'F1-Score': F1Score(name='F1-Score', num_classes=1, dtype=tf.float32),
                               'ROC-AUC': AUC(curve='ROC', name='ROC-AUC'),
                               'PR-AUC': AUC(curve='PR', name='PR-AUC'),
                               'Binary-Accuracy': BinaryAccuracy(name='Binary-Accuracy'),
                               'Precision': Precision(name='Precision'),
                               'Recall': Recall(name='Recall'),
                               'True-Positives': TruePositives(name='True-Positives'),
                               'False-Positives': FalsePositives(name='False-Positives'),
                               'False-Negatives': FalseNegatives(name='False-Negatives'),
                               'True-Negatives': TrueNegatives(name='True-Negatives')}
        
        # Metrics
        self.prediction_type = self.dict_prediction_types[self.project]
        self.loss_name = self.dict_losses_names[self.prediction_type]
        self.loss_function = self.dict_losses_K[self.loss_name]
        self.main_metric_name = self.dict_main_metrics_names_K[self.project]
        self.main_metric_mode = self.main_metrics_modes[self.main_metric_name]
        self.main_metric = self.dict_metrics_K[self.main_metric_name]
        self.metrics_names = [self.main_metric_name]
        self.metrics = [self.dict_metrics_K[metric_name] for metric_name in self.metrics_names]
        
        # Optimizers
        self.optimizers = {'Adam': Adam, 'RMSprop': RMSprop, 'Adadelta': Adadelta}
        
        # Model
        self.model = None
    
    @staticmethod
    def _append_ext(fn):
        return fn + ".jpg"
    
    def _load_data_features(self):
        for fold in self.folds:
            self.DATA_FEATURES[fold] = pd.read_csv(
                self.spath + 'dataframes/data-features_' + self.target + '_' + self.organ + '_' + self.view + '_' + self.transformation + '_' + fold + '_' + self.outer_fold + '.csv')
            for col_name in self.id_vars:
                self.DATA_FEATURES[fold][col_name] = self.DATA_FEATURES[fold][col_name].astype(str)
            self.DATA_FEATURES[fold].set_index('eid', drop=False, inplace=True)
    
    def _take_subset_to_debug(self):
        for fold in self.folds:
            # use +1 or +2 to test the leftovers pipeline
            leftovers_extra = {'train': 0, 'val': 1, 'test': 2}
            n_batches = 2
            n_limit_fold = leftovers_extra[fold] + self.batch_size * n_batches
            self.DATA_FEATURES[fold] = self.DATA_FEATURES[fold].iloc[:n_limit_fold, :]
    
    def _generate_generators(self, DATA_FEATURES):
        GENERATORS = {}
        for fold in self.folds:
            # do not generate a generator if there are no samples (can happen for leftovers generators)
            if fold not in DATA_FEATURES.keys():
                continue
            # parameters
            training_mode = True if self.mode == 'model_training' else False
            if (fold == 'train') & (self.mode == 'model_training') & \
                    (self.organ + '_' + self.view not in self.organsviews_not_to_augment):
                data_augmentation = True
            else:
                data_augmentation = False
            # define batch size for testing: data is split between a part that fits in batches, and leftovers
            if self.mode == 'model_testing':
                if self.organ + '_' + self.view in self.left_right_organs_views:
                    n_samples = len(DATA_FEATURES[fold].index) * 2
                else:
                    n_samples = len(DATA_FEATURES[fold].index)
                batch_size_fold = min(self.batch_size, n_samples)
            else:
                batch_size_fold = self.batch_size
            if (fold == 'train') & (self.mode == 'model_training'):
                n_samples_per_subepoch = self.n_samples_per_subepoch
            else:
                n_samples_per_subepoch = None
            # generator
            GENERATORS[fold] = \
                MyImageDataGenerator(organ=self.organ, view=self.view,
                                     data_features=DATA_FEATURES[fold], n_samples_per_subepoch=n_samples_per_subepoch,
                                     batch_size=batch_size_fold, training_mode=training_mode,
                                     side_predictors=self.side_predictors, dir_images=self.dir_images,
                                     images_width=self.image_width, images_height=self.image_height,
                                     data_augmentation=data_augmentation,
                                     data_augmentation_factor=self.data_augmentation_factor, seed=self.seed)
        return GENERATORS
    
    def _generate_class_weights(self):
        if self.dict_prediction_types[self.project] == 'binary':
            self.class_weights = {}
            counts = self.DATA_FEATURES['train'][self.target + '_raw'].value_counts()
            n_total = counts.sum()
            # weighting the samples for each class inversely proportional to their prevalence, with order of magnitude 1
            for i in counts.index.values:
                self.class_weights[i] = n_total / (counts.loc[i] * len(counts.index))
    
    def _generate_cnn(self):
        # define the arguments
        # take special initial weights for EfficientNetB7 (better)
        if (self.architecture == 'EfficientNetB7') & (self.keras_weights == 'imagenet'):
            w = 'noisy-student'
        else:
            w = self.keras_weights
        kwargs = {"include_top": False, "weights": w, "input_shape": (self.image_width, self.image_height, 3)}
        if self.architecture in ['ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
                                 'ResNeXt50', 'ResNeXt101']:
            import tensorflow.keras
            kwargs.update(
                {"backend": tensorflow.keras.backend, "layers": tensorflow.keras.layers,
                 "models": tensorflow.keras.models, "utils": tensorflow.keras.utils})
        
        # load the architecture builder
        if self.architecture == 'VGG16':
            from tensorflow.keras.applications.vgg16 import VGG16 as ModelBuilder
        elif self.architecture == 'VGG19':
            from tensorflow.keras.applications.vgg19 import VGG19 as ModelBuilder
        elif self.architecture == 'DenseNet121':
            from tensorflow.keras.applications.densenet import DenseNet121 as ModelBuilder
        elif self.architecture == 'DenseNet169':
            from tensorflow.keras.applications.densenet import DenseNet169 as ModelBuilder
        elif self.architecture == 'DenseNet201':
            from tensorflow.keras.applications.densenet import DenseNet201 as ModelBuilder
        elif self.architecture == 'Xception':
            from tensorflow.keras.applications.xception import Xception as ModelBuilder
        elif self.architecture == 'InceptionV3':
            from tensorflow.keras.applications.inception_v3 import InceptionV3 as ModelBuilder
        elif self.architecture == 'InceptionResNetV2':
            from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 as ModelBuilder
        elif self.architecture == 'ResNet50':
            from keras_applications.resnet import ResNet50 as ModelBuilder
        elif self.architecture == 'ResNet101':
            from keras_applications.resnet import ResNet101 as ModelBuilder
        elif self.architecture == 'ResNet152':
            from keras_applications.resnet import ResNet152 as ModelBuilder
        elif self.architecture == 'ResNet50V2':
            from keras_applications.resnet_v2 import ResNet50V2 as ModelBuilder
        elif self.architecture == 'ResNet101V2':
            from keras_applications.resnet_v2 import ResNet101V2 as ModelBuilder
        elif self.architecture == 'ResNet152V2':
            from keras_applications.resnet_v2 import ResNet152V2 as ModelBuilder
        elif self.architecture == 'ResNeXt50':
            from keras_applications.resnext import ResNeXt50 as ModelBuilder
        elif self.architecture == 'ResNeXt101':
            from keras_applications.resnext import ResNeXt101 as ModelBuilder
        elif self.architecture == 'EfficientNetB7':
            from efficientnet.tfkeras import EfficientNetB7 as ModelBuilder
        # The following model have a fixed input size requirement
        elif self.architecture == 'NASNetMobile':
            from tensorflow.keras.applications.nasnet import NASNetMobile as ModelBuilder
        elif self.architecture == 'NASNetLarge':
            from tensorflow.keras.applications.nasnet import NASNetLarge as ModelBuilder
        elif self.architecture == 'MobileNet':
            from tensorflow.keras.applications.mobilenet import MobileNet as ModelBuilder
        elif self.architecture == 'MobileNetV2':
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as ModelBuilder
        else:
            print('Architecture does not exist.')
            sys.exit(1)
        
        # build the model's base
        cnn = ModelBuilder(**kwargs)
        x = cnn.output
        # complete the model's base
        if self.architecture in ['VGG16', 'VGG19']:
            x = Flatten()(x)
            x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            x = Dropout(self.dropout_rate)(x)
            x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            x = Dropout(self.dropout_rate)(x)
        else:
            x = GlobalAveragePooling2D()(x)
            if self.architecture == 'EfficientNetB7':
                x = Dropout(self.dropout_rate)(x)
        cnn_output = x
        return cnn.input, cnn_output
    
    def _generate_side_nn(self):
        side_nn = Sequential()
        side_nn.add(Dense(16, input_dim=len(self.side_predictors), activation="relu",
                          kernel_regularizer=regularizers.l2(self.weight_decay)))
        return side_nn.input, side_nn.output
    
    def _complete_architecture(self, cnn_input, cnn_output, side_nn_input, side_nn_output):
        x = concatenate([cnn_output, side_nn_output])
        x = Dropout(self.dropout_rate)(x)
        l_ns = [int(self.n_fc_nodes * (2 ** (2 * (self.n_fc_layers - 1 - i)))) for i in range(self.n_fc_layers)]
        for n in [int(self.n_fc_nodes * (2 ** (2 * (self.n_fc_layers - 1 - i)))) for i in range(self.n_fc_layers)]:
            x = Dense(n, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            # scale the dropout proportionally to the number of nodes in a layer. No dropout for the last layers
            if n > 16:
                x = Dropout(self.dropout_rate * n / max(l_ns))(x)
        predictions = Dense(1, activation=self.dict_final_activations[self.prediction_type],
                            kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        self.model = Model(inputs=[cnn_input, side_nn_input], outputs=predictions)
    
    def _generate_architecture(self):
        cnn_input, cnn_output = self._generate_cnn()
        side_nn_input, side_nn_output = self._generate_side_nn()
        self._complete_architecture(cnn_input=cnn_input, cnn_output=cnn_output, side_nn_input=side_nn_input,
                                    side_nn_output=side_nn_output)
    
    def _load_model_weights(self):
        try:
            print(self.path_load_weights)
            self.model.load_weights(self.path_load_weights)
        except (FileNotFoundError, TypeError):
            # load backup weights if the main weights are corrupted
            try:
                self.model.load_weights(self.path_load_weights.replace('model-weights', 'backup-model-weights'))
            except FileNotFoundError:
                print('Error. No file was found. imagenet weights should have been used. Bug somewhere.')
                sys.exit(1)
    
#     @staticmethod
#     def clean_exit():
#         # exit
#         print('\nDone.\n')
#         print('Killing JOB PID with kill...')
#         os.system('touch ../eo/' + os.environ['SLURM_JOBID'])
#         os.system('kill ' + str(os.getpid()))
#         time.sleep(60)
#         print('Escalating to kill JOB PID with kill -9...')
#         os.system('kill -9 ' + str(os.getpid()))
#         time.sleep(60)
#         print('Escalating to kill JOB ID')
#         os.system('scancel ' + os.environ['SLURM_JOBID'])
#         time.sleep(60)
#         print('Everything failed to kill the job. Hanging there until hitting walltime...')
        
        
class Training(DeepLearning):
    
    """
    Class to train CNN models:
    - Generates the architecture
    - Loads the best last weights so that a model can be trained over several jobs
    - Generates the callbacks
    - Compiles the model
    - Trains the model
    """
    
    def __init__(self, target= None, organ=None, view=None, transformation=None, architecture=None, n_fc_layers=None,
                 n_fc_nodes=None, optimizer=None, learning_rate=None, weight_decay=None, dropout_rate=None,
                 data_augmentation_factor=None, outer_fold=None, debug_mode=False, taking_alans_weights=False, 
                 transfer_learning=None,continue_training=True, display_full_metrics=True):
        # parameters
        DeepLearning.__init__(self, target, organ, view, transformation, architecture, n_fc_layers, n_fc_nodes,
                              optimizer, learning_rate, weight_decay, dropout_rate, data_augmentation_factor,
                              debug_mode)
        self.outer_fold = outer_fold
        self.version = self.version + '_' + str(outer_fold)
        # NNet's architecture's weights
        self.continue_training = continue_training
        self.transfer_learning = transfer_learning
        self.list_parameters_to_match = ['organ', 'transformation', 'view']
        # dict to decide in which order targets should be used when trying to transfer weight from a similar model
        self.dict_alternative_targets_for_transfer_learning = {'PRS': ['PRS', 'Sex'], 'Sex': ['Sex', 'PRS']}
        
        # Generators
        self.folds = ['train', 'val']
        self.mode = 'model_training'
        self.class_weights = None
        self.GENERATORS = None
        
        # Metrics
        self.baseline_performance = None
        if display_full_metrics:
            self.metrics_names = self.dict_metrics_names_K[self.prediction_type]
            
        #alans
        self.taking_alans_weights = taking_alans_weights
        
        # Model
        self.path_load_weights = local_to_alans(self.spath + 'weights/model-weights_' + self.version + '.h5', self.taking_alans_weights)
        print('paht_load_weights:', self.path_load_weights)
        if debug_mode:
            self.path_save_weights = self.spath + 'weights/' + 'model-weights-debug.h5'
        else:
            self.path_save_weights = self.spath + 'weights/' + 'model-weights_' + self.version + '.h5'
        print('path_save_weights:', self.path_save_weights)
        self.n_epochs_max = 1000
        self.callbacks = None
        
    
    # Load and preprocess the data, build the generators
    def data_preprocessing(self):
        self._load_data_features()
        if self.debug_mode:
            self._take_subset_to_debug()
        self._generate_class_weights()
        self.GENERATORS = self._generate_generators(self.DATA_FEATURES)
    
    # Determine which weights to load, if any.
    def _weights_for_transfer_learning(self):
        print('Looking for models to transfer weights from...')
        
        #for now, loading weights from Age
        
#         if self.taking_alans_weights:
#             self.path_load_weights = self.path_load_weights.replace('_PRS_', '_Age_')
        
        # define parameters
        parameters = self._version_to_parameters(self.version)
        print(parameters)
        
        # continue training if possible
        if self.continue_training and os.path.exists(self.path_load_weights):
            print('Loading the weights from the model\'s previous training iteration.')
            return
        
        # Initialize the weights using other the weights from other successful hyperparameters combinations
        if self.transfer_learning == 'hyperparameters':
            # Check if the same model with other hyperparameters have already been trained. Pick the best for transfer.
            params = self.version.split('_')
            params_tl_idx = \
                [i for i in range(len(names_model_parameters))
                 if any(names_model_parameters[i] == p for p in
                        ['optimizer', 'learning_rate', 'weight_decay', 'dropout_rate', 'data_augmentation_factor'])]
            for idx in params_tl_idx:
                params[idx] = '*'
                
            if taking_alans_weights:
                versions = self.ipath + '../eo/MI02_' + '_'.join(params) + '.out'
            else:
                versions = self.spath + '../eo/MI02_' + '_'.join(params) + '.out'
                
            files = glob.glob(versions)
            if self.main_metric_mode == 'min':
                best_perf = np.Inf
            else:
                best_perf = -np.Inf
            for file in files:
                hand = open(file, 'r')
                # find best last performance
                final_improvement_line = None
                baseline_performance_line = None
                for line in hand:
                    line = line.rstrip()
                    if re.search('Baseline validation ' + self.main_metric_name + ' = ', line):
                        baseline_performance_line = line
                    if re.search('val_' + self.main_metric_name + ' improved from', line):
                        final_improvement_line = line
                hand.close()
                if final_improvement_line is not None:
                    perf = float(final_improvement_line.split(' ')[7].replace(',', ''))
                elif baseline_performance_line is not None:
                    perf = float(baseline_performance_line.split(' ')[-1])
                else:
                    continue
                # Keep track of the file with the best performance
                if self.main_metric_mode == 'min':
                    update = perf < best_perf
                else:
                    update = perf > best_perf
                if update:
                    best_perf = perf
                    self.path_load_weights = \
                        file.replace('../eo/', '../data/').replace('MI02', 'model-weights').replace('.out', '.h5')
            if best_perf not in [-np.Inf, np.Inf]:
                print('Transfering the weights from: ' + self.path_load_weights + ', with ' + self.main_metric_name +
                      ' = ' + str(best_perf))
                return
        
        # Initialize the weights based on models trained on different datasets, ranked by similarity
        if self.transfer_learning == 'datasets':
            while True:
                # print('Matching models for the following criterias:');
                # print(['architecture', 'target'] + list_parameters_to_match)
                # start by looking for models trained on the same target, then move to other targets
                for target_to_load in self.dict_alternative_targets_for_transfer_learning[parameters['target']]:
                    # print('Target used: ' + target_to_load)
                    parameters_to_match = parameters.copy()
                    parameters_to_match['target'] = target_to_load
                    # load the ranked performances table to select the best performing model among the similar
                    # models available
                    if taking_alans_weights:
                        path_performances_to_load = self.ipath + '../data/' + 'PERFORMANCES_ranked_' + \
                                                parameters_to_match['target'] + '_' + 'val' + '.csv'
                    else:
                        path_performances_to_load = self.spath + '../data/' + 'dataframes/PERFORMANCES_ranked_' + \
                                                parameters_to_match['target'] + '_' + 'val' + '.csv'   
                    try:
                        Performances = pd.read_csv(path_performances_to_load)
                        Performances['organ'] = Performances['organ'].astype(str)
                    except FileNotFoundError:
                        # print("Could not load the file: " + path_performances_to_load)
                        break
                    # iteratively get rid of models that are not similar enough, based on the list
                    for parameter in ['architecture', 'target'] + self.list_parameters_to_match:
                        Performances = Performances[Performances[parameter] == parameters_to_match[parameter]]
                    # if at least one model is similar enough, load weights from the best of them
                    if len(Performances.index) != 0:
                        if taking_alans_weights:
                            self.path_load_weights = self.ipath + '../data/' + 'model-weights_' + Performances['version'][0] + '.h5'
                        else:
                            self.path_load_weights = self.spath + '../data/weights/' + 'model-weights_' + Performances['version'][0] + '.h5'
                        self.keras_weights = None
                        print('transfering the weights from: ' + self.path_load_weights)
                        return
                
                # if no similar model was found, try again after getting rid of the last selection criteria
                if len(self.list_parameters_to_match) == 0:
                    print('No model found for transfer learning.')
                    break
                self.list_parameters_to_match.pop()
        # Otherwise use imagenet weights to initialize
        print('Using imagenet weights.')
        # using string instead of None for path to not ge
        self.path_load_weights = None
        self.keras_weights = 'imagenet'
    
    def _compile_model(self):
        # if learning rate was reduced with success according to logger, start with this reduced learning rate
        if self.path_load_weights is not None:
            path_logger = self.path_load_weights.replace('model-weights', 'logger').replace('.h5', '.csv')
        else:
            path_logger = local_to_alans(self.spath + 'weights/' + 'logger_' + self.version + '.csv', self.taking_alans_weights)
        if os.path.exists(path_logger):
            try:
                logger = pd.read_csv(path_logger)
                best_log = \
                    logger[logger['val_' + self.main_metric_name] == logger['val_' + self.main_metric_name].max()]
                lr = best_log['learning_rate'].values[0]
            except pd.errors.EmptyDataError:
                os.remove(path_logger)
                lr = self.learning_rate
        else:
            lr = self.learning_rate
        self.model.compile(optimizer=self.optimizers[self.optimizer](learning_rate=lr, clipnorm=1.0), loss=self.loss_function,
                           metrics=self.metrics)
    
    def _compute_baseline_performance(self):
        # calculate initial val_loss value
        if self.continue_training:
            idx_metric_name = ([self.loss_name] + self.metrics_names).index(self.main_metric_name)
            baseline_perfs = self.model.evaluate(self.GENERATORS['val'], steps=self.GENERATORS['val'].steps)
            self.baseline_performance = baseline_perfs[idx_metric_name]
        elif self.main_metric_mode == 'min':
            self.baseline_performance = np.Inf
        else:
            self.baseline_performance = -np.Inf
        print('Baseline validation ' + self.main_metric_name + ' = ' + str(self.baseline_performance))
    
    def _define_callbacks(self):
        if self.debug_mode:
            path_logger = self.spath + 'weights/' + 'logger-debug.csv'
            append = False
        else:
            path_logger = self.spath + 'weights/logger_' + self.version + '.csv'
            append = self.continue_training
        csv_logger = MyCSVLogger(path_logger, separator=',', append=append)
        model_checkpoint_backup = MyModelCheckpoint(self.path_save_weights.replace('model-weights',
                                                                                   'backup-model-weights'),
                                                    monitor='val_' + self.main_metric.name,
                                                    baseline=self.baseline_performance, verbose=0, save_best_only=True,
                                                    save_weights_only=True, mode=self.main_metric_mode,
                                                    save_freq='epoch')
        model_checkpoint = MyModelCheckpoint(self.path_save_weights,
                                             monitor='val_' + self.main_metric.name, baseline=self.baseline_performance,
                                             verbose=1, save_best_only=True, save_weights_only=True,
                                             mode=self.main_metric_mode, save_freq='epoch')
        patience_reduce_lr = min(7, 3 * self.GENERATORS['train'].n_subepochs_per_epoch)
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1,
                                                 mode='min', min_delta=0, cooldown=0, min_lr=0)

        early_stopping = EarlyStopping(monitor='val_' + self.main_metric.name, min_delta=0, patience=9, verbose=1,
                                       mode=self.main_metric_mode, baseline=self.baseline_performance,
                                       restore_best_weights=True)
        self.callbacks = [csv_logger, model_checkpoint_backup, model_checkpoint, early_stopping, reduce_lr_on_plateau] # early_stopping out
    
        
    def build_model(self):
        self._weights_for_transfer_learning()
        self._generate_architecture()
        # Load weights if possible
        try:
            load_weights = True if os.path.exists(self.path_load_weights) else False
        except TypeError:
            load_weights = False
        if load_weights:
            self._load_model_weights()
        else:
            # save transferred weights as default, in case no better weights are found
            self.model.save_weights(self.path_save_weights.replace('model-weights', 'backup-model-weights'))
            self.model.save_weights(self.path_save_weights)
        self._compile_model()
        self._compute_baseline_performance()
        self._define_callbacks()
    
    def train_model(self):
        # garbage collector
        _ = gc.collect()
        # use more verbose when debugging
        verbose = 1 if self.debug_mode else 2
        
        # train the model
        self.model.fit(self.GENERATORS['train'], steps_per_epoch=self.GENERATORS['train'].steps,
                       validation_data=self.GENERATORS['val'], validation_steps=self.GENERATORS['val'].steps,
                       shuffle=False, use_multiprocessing=True, workers=self.n_cpus, epochs=self.n_epochs_max,
                       class_weight=self.class_weights, callbacks=self.callbacks, verbose=verbose)


        
class PredictionsGenerate(DeepLearning):
    
    """
    Generates the predictions for each model.
    Unscales the predictions.
    """
    
    def __init__(self, target=None, organ=None, view=None, transformation=None, architecture=None, n_fc_layers=None,
                 n_fc_nodes=None, optimizer=None, learning_rate=None, weight_decay=None, dropout_rate=None,
                 data_augmentation_factor=None, outer_fold=None, debug_mode=False):
        # Initialize parameters
        DeepLearning.__init__(self, target, organ, view, transformation, architecture, n_fc_layers, n_fc_nodes,
                              optimizer, learning_rate, weight_decay, dropout_rate, data_augmentation_factor, 
                              debug_mode)
        self.outer_fold = outer_fold
        self.mode = 'model_testing'
        # Define dictionaries attributes for data, generators and predictions
        self.DATA_FEATURES_BATCH = {}
        self.DATA_FEATURES_LEFTOVERS = {}
        self.GENERATORS_BATCH = None
        self.GENERATORS_LEFTOVERS = None
        self.PREDICTIONS = {}
    
    def _split_batch_leftovers(self):
        # split the samples into two groups: what can fit into the batch size, and the leftovers.
        for fold in self.folds:
            n_leftovers = len(self.DATA_FEATURES[fold].index) % self.n_ids_batch
            if n_leftovers > 0:
                self.DATA_FEATURES_BATCH[fold] = self.DATA_FEATURES[fold].iloc[:-n_leftovers]
                self.DATA_FEATURES_LEFTOVERS[fold] = self.DATA_FEATURES[fold].tail(n_leftovers)
            else:
                self.DATA_FEATURES_BATCH[fold] = self.DATA_FEATURES[fold]  # special case for syntax if no leftovers
                if fold in self.DATA_FEATURES_LEFTOVERS.keys():
                    del self.DATA_FEATURES_LEFTOVERS[fold]
    
    def _generate_outerfold_predictions(self):
        # prepare unscaling
        if self.project in self.targets_regression:
            mean_train = self.DATA_FEATURES['train']['target' + '_raw'].mean()
            std_train = self.DATA_FEATURES['train']['target' + '_raw'].std()
        else:
            mean_train, std_train = None, None
        print('mean_train:', mean_train, '\nstd_train:', std_train)

        # Generate predictions
        for fold in self.folds:
            print('Predicting samples from fold ' + fold + '.')
            print(str(len(self.DATA_FEATURES[fold].index)) + ' samples to predict.')
            print('Predicting batches: ' + str(len(self.DATA_FEATURES_BATCH[fold].index)) + ' samples.')
            pred_batch = self.model.predict(self.GENERATORS_BATCH[fold], steps=self.GENERATORS_BATCH[fold].steps,
                                            verbose=1)
            if fold in self.GENERATORS_LEFTOVERS.keys():
                print('Predicting leftovers: ' + str(len(self.DATA_FEATURES_LEFTOVERS[fold].index)) + ' samples.')
                pred_leftovers = self.model.predict(self.GENERATORS_LEFTOVERS[fold],
                                                    steps=self.GENERATORS_LEFTOVERS[fold].steps, verbose=1)
                pred_full = np.concatenate((pred_batch, pred_leftovers)).squeeze()
            else:
                pred_full = pred_batch.squeeze()
            print('Predicted a total of ' + str(len(pred_full)) + ' samples.')
            # take the average between left and right predictions for paired organs
            if self.organ + '_' + self.view in self.left_right_organs_views:
                pred_full = np.mean(pred_full.reshape(-1, 2), axis=1)
            # unscale predictions
            if self.project in self.targets_regression:
                print('de_standardizing data')
                pred_full = pred_full * std_train + mean_train
            # format the dataframe
            self.DATA_FEATURES[fold]['pred'] = pred_full
            self.PREDICTIONS[fold] = self.DATA_FEATURES[fold]
            self.PREDICTIONS[fold]['eid'] = [ID.replace('.jpg', '') for ID in self.PREDICTIONS[fold]['eid']]
    
    def _generate_predictions(self):
        if self.debug_mode == True:
            self.path_load_weights = self.spath + 'weights/model-weights-debug.h5'
        else:
            self.path_load_weights = self.spath + 'weights/model-weights_' + self.version + '_' + self.outer_fold + '.h5'

        self._load_data_features()
        if self.debug_mode:
            self._take_subset_to_debug()
        self._load_model_weights()
        self._split_batch_leftovers()
        # generate the generators
        self.GENERATORS_BATCH = self._generate_generators(DATA_FEATURES=self.DATA_FEATURES_BATCH)
        if self.DATA_FEATURES_LEFTOVERS is not None:
            self.GENERATORS_LEFTOVERS = self._generate_generators(DATA_FEATURES=self.DATA_FEATURES_LEFTOVERS)
        self._generate_outerfold_predictions()
    
    def _format_predictions(self):
        for fold in self.folds:
            perf_fun = self.dict_metrics_sklearn[self.dict_main_metrics_names[self.project]]
            print('the metric is:', str(perf_fun))
            perf = perf_fun(self.PREDICTIONS[fold]['target' + '_raw'], self.PREDICTIONS[fold]['pred'])
            print('The ' + fold + ' performance is: ' + str(perf))
            # format the predictions
            self.PREDICTIONS[fold].index.name = 'column_names'
            self.PREDICTIONS[fold] = self.PREDICTIONS[fold][['eid', 'outer_fold', 'pred']]
    
    def generate_predictions(self):
        self._generate_architecture()
        self._generate_predictions()
        self._format_predictions()
    
    def save_predictions(self):
        for fold in self.folds:
            self.PREDICTIONS[fold].to_csv(self.spath + 'dataframes/Predictions_' + self.version + '_' + fold + '_'
                                          + self.outer_fold + '.csv', index=False)
        
        
        
class PredictionsConcatenate(Basics):
    
    """
    Concatenates the predictions coming from the different cross validation folds.
    """
    
    def __init__(self, target = None, organ=None, view=None, transformation=None, architecture=None, n_fc_layers=None,
                 n_fc_nodes=None, optimizer=None, learning_rate=None, weight_decay=None, dropout_rate=None,
                 data_augmentation_factor=None):
        # Initialize parameters
        Basics.__init__(self)
        self.version = target + '_' + organ + '_' + view + '_' + transformation + '_' + architecture + '_' + \
                       n_fc_layers + '_' + n_fc_nodes + '_' + optimizer + '_' + learning_rate + '_' + weight_decay + \
                       '_' + dropout_rate + '_' + data_augmentation_factor
        # Define dictionaries attributes for data, generators and predictions
        self.PREDICTIONS = {}
    
    def concatenate_predictions(self):
        for fold in self.folds:
            for outer_fold in self.outer_folds:
                print('path:', self.spath + 'dataframes/Predictions_' + self.version + '_' + fold + '_' + outer_fold + '.csv')
                if os.path.exists(self.spath + 'dataframes/Predictions_' + self.version + '_' + fold +
                                               '_' + outer_fold + '.csv'):
                    Predictions_fold = pd.read_csv(self.spath + 'dataframes/Predictions_' + self.version + '_' + fold +
                                                   '_' + outer_fold + '.csv')
                    if fold in self.PREDICTIONS.keys():
                        self.PREDICTIONS[fold] = pd.concat([self.PREDICTIONS[fold], Predictions_fold])
                    else:
                        self.PREDICTIONS[fold] = Predictions_fold
                else:
                    print(f'MISSING outer_folders for prediction concatenation for fold {fold} and outer_fold {outer_fold}!')
    
    def save_predictions(self):
        for fold in self.folds:
            print(self.spath + 'dataframes/Predictions_' + self.version + '_' + fold + '.csv')
            self.PREDICTIONS[fold].to_csv(self.spath + 'dataframes/Predictions_' + self.version + '_' + fold + '.csv', index=False)
            
            
            
class PredictionsMerge(Basics):
    
    """
    Merges the predictions from all models into a unified dataframe.
    """
    
    def __init__(self, target=None, fold=None):
        
        Basics.__init__(self)
        
        # Define dictionaries attributes for data, generators and predictions
        self.target = target
        self.fold = fold
        self.data_features = None
        self.list_models = None
        self.Predictions_df_previous = None
        self.Predictions_df = None
    
    def _load_data_features(self):
        self.data_features = pd.read_csv(self.spath + f'dataframes/data-features_computed_{self.target}.csv',
                                         usecols=self.id_vars + self.demographic_vars)
        for var in self.id_vars:
            self.data_features[var] = self.data_features[var].astype(str)
        self.data_features.set_index('eid', drop=False, inplace=True)
        self.data_features.index.name = 'column_names'
    
    def _preprocess_data_features(self):
        # For the training set, each sample is predicted n_CV_outer_folds times, so prepare a larger dataframe
        if self.fold == 'train':
            df_all_folds = None
            for outer_fold in self.outer_folds:
                df_fold = self.data_features.copy()
                df_all_folds = df_fold if outer_fold == self.outer_folds[0] else df_all_folds.append(df_fold)
            self.data_features = df_all_folds
    
    def _list_models(self):
        # generate list of predictions that will be integrated in the Predictions dataframe
        self.list_models = glob.glob(self.spath + 'dataframes/Predictions_' + self.target + '_*_' + self.fold + '.csv')
        # get rid of ensemble models and models already merged
        self.list_models = [model for model in self.list_models if ('*' not in model)]
        # if self.Predictions_df_previous is not None:
        #     self.list_models = \
        #         [model for model in self.list_models
        #          if ('pred_' + '_'.join(model.split('_')[2:-1]) not in self.Predictions_df_previous.columns)]
        self.list_models.sort()
    
    def preprocessing(self):
        self._load_data_features()
        self._preprocess_data_features()
        # self._load_previous_merged_predictions()
        self._list_models()
    
    def merge_predictions(self):
        # merge the predictions
        print('There are ' + str(len(self.list_models)) + ' models to merge.')
        i = 0
        # define subgroups to accelerate merging process
        list_subgroups = list(set(['_'.join(model.split('_')[3:7]) for model in self.list_models]))
        for subgroup in list_subgroups:
            print('Merging models from the subgroup ' + subgroup)
            models_subgroup = [model for model in self.list_models if subgroup in model]
            Predictions_subgroup = None
            # merge the models one by one
            for file_name in models_subgroup:
                i += 1
                version = '_'.join(file_name.split('_')[1:-1])
                # if self.Predictions_df_previous is not None and \
                #         'pred_' + version in self.Predictions_df_previous.columns:
                #     print('The model ' + version + ' has already been merged before.')
                # else:
                print('Merging the ' + str(i) + 'th model: ' + version)
                # load csv and format the predictions
                prediction = pd.read_csv(file_name)
                print('raw prediction\'s shape: ' + str(prediction.shape))
                for var in ['eid', 'outer_fold']:
                    prediction[var] = prediction[var].apply(str)
                prediction.rename(columns={'pred': 'pred_' + version}, inplace=True)
                # merge data frames
                if Predictions_subgroup is None:
                    Predictions_subgroup = prediction
                elif self.fold == 'train':
                    Predictions_subgroup = Predictions_subgroup.merge(prediction, how='outer',
                                                                      on=['eid', 'outer_fold'])
                else:
                    prediction.drop(['outer_fold'], axis=1, inplace=True)
                    # not supported for panda version > 0.23.4 for now
                    Predictions_subgroup = Predictions_subgroup.merge(prediction, how='outer', on=['eid'])
            
            # merge group predictions data frames
            if self.fold != 'train':
                Predictions_subgroup.drop(['outer_fold'], axis=1, inplace=True)
            if Predictions_subgroup is not None:
                if self.Predictions_df is None:
                    self.Predictions_df = Predictions_subgroup
                elif self.fold == 'train':
                    self.Predictions_df = self.Predictions_df.merge(Predictions_subgroup, how='outer',
                                                                    on=['eid', 'outer_fold'])
                else:
                    # not supported for panda version > 0.23.4 for now
                    self.Predictions_df = self.Predictions_df.merge(Predictions_subgroup, how='outer', on=['eid'])
                print('Predictions_df\'s shape: ' + str(self.Predictions_df.shape))
                # garbage collector
                gc.collect()
        
        # # Merge with the previously merged predictions
        # if (self.Predictions_df_previous is not None) & (self.Predictions_df is not None):
        #     if self.fold == 'train':
        #         self.Predictions_df = self.Predictions_df_previous.merge(self.Predictions_df, how='outer',
        #                                                                  on=['id', 'outer_fold'])
        #     else:
        #         self.Predictions_df.drop(columns=['outer_fold'], inplace=True)
        #         # not supported for panda version > 0.23.4 for now
        #         self.Predictions_df = self.Predictions_df_previous.merge(self.Predictions_df, how='outer', on=['id'])
        #     self.Predictions_df_previous = None
        # elif self.Predictions_df is None:
        #     print('No new models to merge. Exiting.')
        #     print('Done.')
        #     sys.exit(0)
        
        # Reorder the columns alphabetically
        pred_versions = [col for col in self.Predictions_df.columns if 'pred_' in col]
        pred_versions.sort()
        id_cols = ['eid', 'outer_fold'] if self.fold == 'train' else ['eid']
        self.Predictions_df = self.Predictions_df[id_cols + pred_versions]
    
    def postprocessing(self):
        # get rid of useless rows in data_features before merging to keep the memory requirements as low as possible
        self.data_features = self.data_features[self.data_features['eid'].isin(self.Predictions_df['eid'].values)]
        # merge data_features and predictions
        if self.fold == 'train':
            print('Starting to merge a massive dataframe')
            self.Predictions_df = self.data_features.merge(self.Predictions_df, how='outer', on=['eid', 'outer_fold'])
        else:
            # not supported for panda version > 0.23.4 for now
            self.Predictions_df = self.data_features.merge(self.Predictions_df, how='outer', on=['eid'])
        print('Merging done')
        
        # remove rows for which no prediction is available (should be none)
        subset_cols = [col for col in self.Predictions_df.columns if 'pred_' in col]
        self.Predictions_df.dropna(subset=subset_cols, how='all', inplace=True)
        
        # Displaying the R2s
        versions = [col.replace('pred_', '') for col in self.Predictions_df.columns if 'pred_' in col]
        r2s = []
        for version in versions:
            df = self.Predictions_df[['target', 'pred_' + version]].dropna()
            r2s.append(r2_score(df['target'], df['pred_' + version]))
        R2S = pd.DataFrame({'version': versions, 'R2': r2s})
        R2S.sort_values(by='R2', ascending=False, inplace=True)
        print('R2 for each model: ')
        print(R2S)
    
    def save_merged_predictions(self):
        print('Writing the merged predictions...')
        self.Predictions_df.to_csv(self.spath + 'dataframes/PREDICTIONS_withoutEnsembles_' + self.target + '_' + self.fold + '.csv', index=False)
        
class PerformancesGenerate(Metrics):
    
    """
    Computes the performances for each model.
    """
    
    def __init__(self, target=None, organ=None, view=None, transformation=None, architecture=None, n_fc_layers=None,
                 n_fc_nodes=None, optimizer=None, learning_rate=None, weight_decay=None, dropout_rate=None,
                 data_augmentation_factor=None, fold=None, debug_mode=False):
        
        Metrics.__init__(self)
        self.target = target
        self.organ = organ
        self.view = view
        self.transformation = transformation
        self.architecture = architecture
        self.n_fc_layers = n_fc_layers
        self.n_fc_nodes = n_fc_nodes
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.data_augmentation_factor = data_augmentation_factor
        self.fold = fold
        if debug_mode:
            self.n_bootstrap_iterations = 3
        else:
            self.n_bootstrap_iterations = 1000
        self.version = target + '_' + organ + '_' + view + '_' + transformation + '_' + architecture + '_' + \
                       n_fc_layers + '_' + n_fc_nodes + '_' + optimizer + '_' + learning_rate + '_' + weight_decay + \
                       '_' + dropout_rate + '_' + data_augmentation_factor
        self.names_metrics = self.dict_metrics_names[self.dict_prediction_types[self.project]]
        self.data_features = None
        self.Predictions = None
        self.PERFORMANCES = None
   
    
    def _preprocess_data_features_predictions_for_performances(self):
        # load dataset
        data_features = pd.read_csv(self.spath + 'dataframes/data-features_computed' + '_' + self.target + '.csv',
                                    usecols=['eid', 'target'])
        # format data_features to extract y
        data_features.rename(columns={'target': 'y'}, inplace=True)
        data_features = data_features[['eid', 'y']]
        data_features['eid'] = data_features['eid'].astype(str)
        data_features.set_index('eid', drop=False, inplace=True)
        data_features.index.name = 'columns_names'
        self.data_features = data_features
    
    def _preprocess_predictions_for_performances(self):
        print('version:', self.version)
        Predictions = pd.read_csv(self.spath + 'dataframes/Predictions_' + self.version + '_' +
                                  self.fold + '.csv')
        Predictions['eid'] = Predictions['eid'].astype(str)
        self.Predictions = Predictions.merge(self.data_features, how='inner', on=['eid'])
    
    # Initialize performances dataframes and compute sample sizes
    def _initiate_empty_performances_df(self):
        # Define an empty performances dataframe to store the performances computed
        row_names = ['all'] + self.outer_folds
        col_names_sample_sizes = ['N']
        col_names = ['outer_fold'] + col_names_sample_sizes
        col_names.extend(self.names_metrics)
        performances = np.empty((len(row_names), len(col_names),))
        performances.fill(np.nan)
        performances = pd.DataFrame(performances)
        performances.index = row_names
        performances.columns = col_names
        performances['outer_fold'] = row_names
        # Convert float to int for sample sizes and some metrics.
        for col_name in col_names_sample_sizes:
            # need recent version of pandas to use type below. Otherwise nan cannot be int
            performances[col_name] = performances[col_name].astype('Int64')
        
        # compute sample sizes for the data frame
        performances.loc['all', 'N'] = len(self.Predictions.index)
        for outer_fold in self.outer_folds:
            performances.loc[outer_fold, 'N'] = len(
                self.Predictions.loc[self.Predictions['outer_fold'] == int(outer_fold)].index)
        
        # initialize the dataframes
        self.PERFORMANCES = {}
        for mode in self.modes:
            self.PERFORMANCES[mode] = performances.copy()
        
        # Convert float to int for sample sizes and some metrics.
        for col_name in self.PERFORMANCES[''].columns.values:
            if any(metric in col_name for metric in self.metrics_displayed_in_int):
                # need recent version of pandas to use type below. Otherwise nan cannot be int
                self.PERFORMANCES[''][col_name] = self.PERFORMANCES[''][col_name].astype('Int64')
    
    def preprocessing(self):
        self._preprocess_data_features_predictions_for_performances()
        self._preprocess_predictions_for_performances()
        self._initiate_empty_performances_df()
    
    # Fill the columns for this model, outer_fold by outer_fold
    def compute_performances(self):
        
        # fill it outer_fold by outer_fold
        for outer_fold in ['all'] + self.outer_folds:
            print('Calculating the performances for the outer fold ' + outer_fold)
            # Generate a subdataframe from the predictions table for each outerfold
            if outer_fold == 'all':
                predictions_fold = self.Predictions.copy()
            else:
                predictions_fold = self.Predictions.loc[self.Predictions['outer_fold'] == int(outer_fold), :]
            
            # if no samples are available for this fold, fill columns with nans
            if len(predictions_fold.index) == 0:
                print('NO SAMPLES AVAILABLE FOR MODEL ' + self.version + ' IN OUTER_FOLD ' + outer_fold)
            else:
                predictions_fold_class = None
                # Fill the Performances dataframe metric by metric
                for name_metric in self.names_metrics:
                    # print('Calculating the performance using the metric ' + name_metric)
                    if name_metric in self.metrics_needing_classpred:
                        predictions_metric = predictions_fold_class
                    else:
                        predictions_metric = predictions_fold
                    metric_function = self.dict_metrics_sklearn[name_metric]
                    self.PERFORMANCES[''].loc[outer_fold, name_metric] = metric_function(predictions_metric['y'],
                                                                                         predictions_metric['pred'])
                    self.PERFORMANCES['_sd'].loc[outer_fold, name_metric] = \
                        self._bootstrap(predictions_metric, metric_function)[1]
                    self.PERFORMANCES['_str'].loc[outer_fold, name_metric] = "{:.3f}".format(
                        self.PERFORMANCES[''].loc[outer_fold, name_metric]) + '+-' + "{:.3f}".format(
                        self.PERFORMANCES['_sd'].loc[outer_fold, name_metric])
        
        # calculate the fold sd (variance between the metrics values obtained on the different folds)
        folds_sd = self.PERFORMANCES[''].iloc[1:, :].std(axis=0)
        for name_metric in self.names_metrics:
            self.PERFORMANCES['_str'].loc['all', name_metric] = "{:.3f}".format(
                self.PERFORMANCES[''].loc['all', name_metric]) + '+-' + "{:.3f}".format(
                folds_sd[name_metric]) + '+-' + "{:.3f}".format(self.PERFORMANCES['_sd'].loc['all', name_metric])
        
        # print the performances
        print('Performances for model ' + self.version + ': ')
        print(self.PERFORMANCES['_str'])
    
    def save_performances(self):
        for mode in self.modes:
            path_save = self.spath + 'dataframes/Performances_' + self.version + '_' + self.fold + \
                        mode + '.csv'
            self.PERFORMANCES[mode].to_csv(path_save, index=False)
        
        
class PerformancesMerge(Metrics):
    
    """
    Merges the performances of the different models into a unified dataframe.
    """
    
    def __init__(self, target=None, fold=None, ensemble_models=None):
        
        # Parameters
        Metrics.__init__(self)
        self.target = target
        self.fold = fold
        self.ensemble_models = self.convert_string_to_boolean(ensemble_models)
        self.names_metrics = self.dict_metrics_names[self.dict_prediction_types[self.project]]
        self.main_metric_name = self.dict_main_metrics_names[self.project]
        # list the models that need to be merged
        self.list_models = glob.glob(self.spath + 'dataframes/Performances_' + self.target + '_*_' + fold +
                                     '_str.csv')
        print(self.spath + 'dataframes/Performances_' + self.target + '_*_' + fold +'_str.csv')
        # get rid of ensemble models
        if self.ensemble_models:
            print('ensemble_models:', self.ensemble_models)
            print(self.list_models)
            self.list_models = [model for model in self.list_models if '*' in model]
        else:
            self.list_models = [model for model in self.list_models if '*' not in model]
        self.Performances = None
        self.Performances_alphabetical = None
        self.Performances_ranked = None
    
    def _initiate_empty_performances_summary_df(self):
        # Define the columns of the Performances dataframe
        # columns for sample sizes
        names_sample_sizes = ['N']
        
        # columns for metrics
        names_metrics = self.dict_metrics_names[self.dict_prediction_types[self.project]]
        # for normal folds, keep track of metric and bootstrapped metric's sd
        names_metrics_with_sd = []
        for name_metric in names_metrics:
            names_metrics_with_sd.extend([name_metric, name_metric + '_sd', name_metric + '_str'])
        
        # for the 'all' fold, also keep track of the 'folds_sd' (metric's sd calculated using the folds' results)
        names_metrics_with_folds_sd_and_sd = []
        for name_metric in names_metrics:
            names_metrics_with_folds_sd_and_sd.extend(
                [name_metric, name_metric + '_folds_sd', name_metric + '_sd', name_metric + '_str'])
        
        # merge all the columns together. First description of the model, then sample sizes and metrics for each fold
        names_col_Performances = ['version'] + self.names_model_parameters
        # special outer fold 'all'
        names_col_Performances.extend(
            ['_'.join([name, 'all']) for name in names_sample_sizes + names_metrics_with_folds_sd_and_sd])
        # other outer_folds
        for outer_fold in self.outer_folds:
            names_col_Performances.extend(
                ['_'.join([name, outer_fold]) for name in names_sample_sizes + names_metrics_with_sd])
        
        # Generate the empty Performance table from the rows and columns.
        Performances = np.empty((len(self.list_models), len(names_col_Performances),))
        Performances.fill(np.nan)
        Performances = pd.DataFrame(Performances)
        Performances.columns = names_col_Performances
        # Format the types of the columns
        for colname in Performances.columns.values:
            if (colname in self.names_model_parameters) | ('_str' in colname):
                col_type = str
            else:
                col_type = float
            Performances[colname] = Performances[colname].astype(col_type)
        self.Performances = Performances
    
    def merge_performances(self):
        # define parameters
        names_metrics = self.dict_metrics_names[self.dict_prediction_types[self.project]]
        
        # initiate dataframe
        self._initiate_empty_performances_summary_df()
        
        # Fill the Performance table row by row
        for i, model in enumerate(self.list_models):
            # load the performances subdataframe
            PERFORMANCES = {}
            for mode in self.modes:
                PERFORMANCES[mode] = pd.read_csv(model.replace('_str', mode))
                PERFORMANCES[mode].set_index('outer_fold', drop=False, inplace=True)
            
            # Fill the columns corresponding to the model's parameters
            version = '_'.join(model.split('_')[1:-2])
            print(version)
            parameters = self._version_to_parameters(version)
            
            # fill the columns for model parameters
            self.Performances['version'][i] = version
            for parameter_name in self.names_model_parameters:
                self.Performances[parameter_name][i] = parameters[parameter_name]
            
            # Fill the columns for this model, outer_fold by outer_fold
            for outer_fold in ['all'] + self.outer_folds:
                # Generate a subdataframe from the predictions table for each outerfold
                
                # Fill sample size columns
                self.Performances['N_' + outer_fold][i] = PERFORMANCES[''].loc[outer_fold, 'N']
                
                # Fill the Performances dataframe metric by metric
                for name_metric in names_metrics:
                    for mode in self.modes:
                        self.Performances[name_metric + mode + '_' + outer_fold][i] = PERFORMANCES[mode].loc[
                            outer_fold, name_metric]
                
                # calculate the fold sd (variance between the metrics values obtained on the different folds)
                folds_sd = PERFORMANCES[''].iloc[1:, :].std(axis=0)
                for name_metric in names_metrics:
                    self.Performances[name_metric + '_folds_sd_all'] = folds_sd[name_metric]
        
        # Convert float to int for sample sizes and some metrics.
        for name_col in self.Performances.columns.values:
            cond1 = name_col.startswith('N_')
            cond2 = any(metric in name_col for metric in self.metrics_displayed_in_int)
            cond3 = '_sd' not in name_col
            cond4 = '_str' not in name_col
            if cond1 | cond2 & cond3 & cond4:
                self.Performances[name_col] = self.Performances[name_col].astype('Int64')
                # need recent version of pandas to use this type. Otherwise nan cannot be int
        
        # For ensemble models, merge the new performances with the previously computed performances
        if self.ensemble_models:
            Performances_withoutEnsembles = pd.read_csv(self.spath + 'dataframes/PERFORMANCES_tuned_alphabetical_' + self.target + '_' + self.fold + '.csv')
            self.Performances = Performances_withoutEnsembles.append(self.Performances)
            # reorder the columns (weird: automatic alphabetical re-ordering happened when append was called for 'val')
            self.Performances = self.Performances[Performances_withoutEnsembles.columns]
        
        # Ranking, printing and saving
        self.Performances_alphabetical = self.Performances.sort_values(by='version')
        cols_to_print = ['version', self.main_metric_name + '_str_all']
        print('Performances of the models ranked by models\'names:')
        print(self.Performances_alphabetical[cols_to_print])
        sort_by = self.dict_main_metrics_names[self.project] + '_all'
        sort_ascending = self.main_metrics_modes[self.dict_main_metrics_names[self.project]] == 'min'
        self.Performances_ranked = self.Performances.sort_values(by=sort_by, ascending=sort_ascending)
        print('Performances of the models ranked by the performance on the main metric on all the samples:')
        print(self.Performances_ranked[cols_to_print])
    
    def save_performances(self):
        name_extension = 'withEnsembles' if self.ensemble_models else 'withoutEnsembles'
        path = self.spath + 'dataframes/PERFORMANCES_' + name_extension + '_alphabetical_' + \
               self.target + '_' + self.fold + '.csv'
        self.Performances_alphabetical.to_csv(path, index=False)
        self.Performances_ranked.to_csv(path.replace('_alphabetical_', '_ranked_'), index=False)
        

        

class AttentionMaps(DeepLearning):
    
    """
    Computes the attention maps (saliency maps and Grad_RAM maps) for all images
    """
    
    def __init__(self, d_targets=None, organ=None, view=None, transformation=None, N_samples_attentionmaps=500, debug_mode=False, 
                 only_guided_gradcam=False, load_eids_from_version=False, regenerate_saliencies=True):
        # Partial initialization with placeholders to get access to parameters and functions
        DeepLearning.__init__(self, list(d_targets.keys())[0], organ , view, transformation, 'InceptionResNetV2', '1', '1024', 'Adam',
                              '0.0001', '0.1', '0.5', '1.0', False)
        # Parameters
        self.target = None
        self.version = None
        self.parameters = None
        self.l_outer_folds = None
        
        self.l_eids = []
        self.d_data_features = {}
        self.d_versions = {}
        self.d_parameters = {}
        
        self.l_targets = list(d_targets.keys())
        self.organ = organ
        self.view = view
        self.transformation = transformation
        self.d_targets = d_targets
        
        self.leftright = True if self.organ + '_' + self.view in self.left_right_organs_views else False
        
        self.image_width = None
        self.image_height = None
        self.batch_size = None
        self.N_samples_attentionmaps = N_samples_attentionmaps  # needs to be > 1 for the script to work
        
        if debug_mode:
            self.N_samples_attentionmaps = 2
        self.dir_images = self.ipath + '../images/' + organ + '/' + view + '/' + transformation + '/'
        self.prediction_type = self.dict_prediction_types[self.project]
        
        self.image = None
        self.generator = None
        self.dict_architecture_to_last_conv_layer_name = \
            {'VGG16': 'block5_conv3', 'VGG19': 'block5_conv4', 'MobileNet': 'conv_pw_13_relu',
             'MobileNetV2': 'out_relu', 'DenseNet121': 'relu', 'DenseNet169': 'relu', 'DenseNet201': 'relu',
             'NASNetMobile': 'activation_1136', 'NASNetLarge': 'activation_1396', 'Xception': 'block14_sepconv2_act',
             'InceptionV3': 'mixed10', 'InceptionResNetV2': 'conv_7b_ac', 'EfficientNetB7': 'top_activation' ,'ResNet50':'conv5_block3_out'}
        self.last_conv_layer = None
        self.organs_views_transformations_images = \
            ['Brain_MRI_SagittalRaw', 'Brain_MRI_SagittalReference', 'Brain_MRI_CoronalRaw',
             'Brain_MRI_CoronalReference', 'Brain_MRI_TransverseRaw', 'Brain_MRI_TransverseReference',
             'Eyes_Fundus_Raw', 'Eyes_OCT_Raw', 'Arterial_Carotids_Mixed', 'Arterial_Carotids_LongAxis',
             'Arterial_Carotids_CIMT120', 'Arterial_Carotids_CIMT150', 'Arterial_Carotids_ShortAxis',
             'Heart_MRI_2chambersRaw', 'Heart_MRI_2chambersContrast', 'Heart_MRI_3chambersRaw',
             'Heart_MRI_3chambersContrast', 'Heart_MRI_4chambersRaw', 'Heart_MRI_4chambersContrast',
             'Abdomen_Liver_Raw', 'Abdomen_Liver_Contrast', 'Abdomen_Pancreas_Raw', 'Abdomen_Pancreas_Contrast',
             'Musculoskeletal_Spine_Sagittal', 'Musculoskeletal_Spine_Coronal', 'Musculoskeletal_Hips_MRI',
             'Musculoskeletal_Knees_MRI', 'Musculoskeletal_FullBody_Mixed', 'Musculoskeletal_FullBody_Figure',
             'Musculoskeletal_FullBody_Skeleton', 'Musculoskeletal_FullBody_Flesh',
             'PhysicalActivity_FullWeek_GramianAngularField1minDifference',
             'PhysicalActivity_FullWeek_GramianAngularField1minSummation',
             'PhysicalActivity_FullWeek_MarkovTransitionField1min', 'PhysicalActivity_FullWeek_RecurrencePlots1min']
        self.dict_sexes_to_values = {'Male': 1, 'Female': 0}

        self.only_guided_gradcam = only_guided_gradcam
        self.load_eids_from_version = load_eids_from_version
        self.regenerate_saliencies = regenerate_saliencies
    
    def _select_best_models(self):
        print('_select_best_models')
        # Pick the best model based on the performances
        for target in self.l_targets:
            print('Target:',target)
            path_perf = self.spath + 'dataframes/PERFORMANCES_withoutEnsembles_ranked_' + target + '_test.csv'
            Performances = pd.read_csv(path_perf).set_index('version', drop=False)
            Performances = Performances[(Performances['organ'] == self.organ)
                                        & (Performances['view'] == self.view)
                                        & (Performances['transformation'] == self.transformation)]

            Performances_tmp = Performances
            requested_parameters = self.d_targets[target]
            if requested_parameters != None:
                for parameter in requested_parameters:
                    if parameter in Performances_tmp.columns.tolist():
                        value = requested_parameters[parameter]
                        Performances_tmp = Performances_tmp.loc[Performances_tmp[parameter] == value]
                    else:
                        print(f'{parameter} is not in Performances! So it wont be applied')
                if len(Performances_tmp) == 0:
                    print('There is no set of parameters like requested, the best set of parameters will be chosen!')
                    version = Performances['version'].values[0]
                else:
                    version = Performances_tmp['version'].values[0]
            else:
                version = Performances_tmp['version'].values[0]
            print('  - version:', version)
            self.d_versions[target] = version
            print(target, ': ', version)
            del Performances, Performances_tmp
            # other parameters
            self.d_parameters[target] = self._version_to_parameters(version)

    def _select_outer_folds(self):
        print('_select_outer_folds')
        l_outer_folds = []
        for target in self.l_targets:
            version = self.d_versions[target]
            data_performances = pd.read_csv('../data/dataframes/Performances_' + version + '_test.csv')
            # take all outer fold exept for the all column
            l_outer_folds.append(data_performances.dropna()['outer_fold'].tolist()[1:])
        intersection = list(set(l_outer_folds[0]).intersection(*l_outer_folds[1:]))
        self.l_outer_folds = intersection
        print(f'Intersection in outerfolds:{self.l_outer_folds}')
        
    def _select_eids(self):
        print('_select_eids')
        l_eids = []
        d_eids = {'Male':[], 'Female':[]}
        
        # for each target and each sex, get the eids
        for target in self.l_targets:
            # load eids for target
            data_features = pd.read_csv(f'../data/dataframes/data-features_computed_{target}.csv')
            for sex in ['Male', 'Female']:
                if self.regenerate_saliencies == False:
                    sub_version = '_'.join(self.d_versions[target].split('_')[-8:])
                    activations_path = '../figures/Attention_Maps/' + target + '/' + self.organ + '/' + self.view + '/' + self.transformation + '/' \
                                        + sub_version + '/' + sex + '/'
                    l_generated_eids = list(set([int(file.split('_')[-1].split('.')[0]) for file in os.listdir(activations_path)]))
                # get eids for testing for Male and Female
                data_features['Sex'] = data_features['Sex'].astype(int)
                data_features_sex = data_features.loc[data_features['Sex'] == self.dict_sexes_to_values[sex]]
                l_files = list(set([file[:-6] for file in os.listdir(self.dir_images)]))
                l_eid_image = [int(eid) for eid in l_files]
                if self.regenerate_saliencies == False:
                    l_eid_image = [eid for eid in l_eid_image if eid not in l_generated_eids]
                data_features_sex = data_features_sex.loc[data_features_sex['eid'].isin(l_eid_image)]
                data_features_sex = data_features_sex.loc[data_features_sex['outer_fold'].isin([int(i) for i in self.l_outer_folds])]
                d_eids[sex].append(data_features_sex['eid'].tolist())
        
        # take the intersections of eids through targets
        if not self.load_eids_from_version:
            for sex in ['Male', 'Female']:
                intersection = list(set(d_eids[sex][0]).intersection(*d_eids[sex][1:]))
                random.shuffle(intersection)
                intersection = intersection[:self.N_samples_attentionmaps]
                l_eids += intersection
            self.l_eids = l_eids
        else:
            parameters_eids = self._version_to_parameters(self.load_eids_from_version)
            sub_version = '_'.join(self.load_eids_from_version.split('_')[-8:])
            path_eids = '../figures/Attention_Maps/' + parameters_eids['target'] + '/' + parameters_eids['organ'] + '/' + parameters_eids['view'] + '/' + parameters_eids['transformation'] + '/' \
                                + sub_version + '/'
            for sex in ['Male', 'Female']:
                l_eids_to_load_sex = list(set([int(file.split('_')[-1].split('.')[0]) for file in os.listdir(path_eids + sex)]))
                print(f'{sex} to load:', len(l_eids_to_load_sex))
                intersection = list(set(l_eids_to_load_sex).intersection(*d_eids[sex]))
                print(len(intersection))
                random.shuffle(intersection)
                intersection = intersection[:self.N_samples_attentionmaps]
                l_eids += intersection
            self.l_eids = l_eids
            
        print('Nb eids:', len(l_eids))
        
    def _select_samples(self):
        print('_select_samples')
        inv_dict_sexes_to_values = {v: k for k, v in self.dict_sexes_to_values.items()}
        for target in self.l_targets:
            parameters = self.d_parameters[target]
            if self.organ + '_' + self.view + '_' + self.transformation in self.organs_views_transformations_images:
                DeepLearning.__init__(self, parameters['target'], parameters['organ'], parameters['view'],
                                      parameters['transformation'], parameters['architecture'],
                                      parameters['n_fc_layers'], parameters['n_fc_nodes'],
                                      parameters['optimizer'], parameters['learning_rate'],
                                      parameters['weight_decay'], parameters['dropout_rate'],
                                      parameters['data_augmentation_factor'], False)
                print(parameters['architecture'])
            
            data_features = pd.read_csv(f'../data/dataframes/data-features_computed_{target}.csv')
            data_features = data_features.loc[data_features['eid'].isin(self.l_eids)]
            data_features_sampled = None

            for sex in ['Male', 'Female']:
                data_features['Sex'] = data_features['Sex'].astype(int)
                data_features_sex = data_features.loc[data_features['Sex'] == self.dict_sexes_to_values[sex]]
                l_eids_sex = data_features_sex['eid'].tolist()
                np.random.shuffle(l_eids_sex)
                l_eids_sex = l_eids_sex[:self.N_samples_attentionmaps]
                data_features_sex = data_features_sex.loc[data_features_sex['eid'].isin(l_eids_sex)]
                if data_features_sampled is None:
                    data_features_sampled = data_features_sex
                else:
                    data_features_sampled = data_features_sampled.append(data_features_sex)
    
            # removing organs and target from version
            sub_version = '_'.join(self.d_versions[self.target].split('_')[-8:])

            activations_path = '../figures/Attention_Maps/' + self.target + '/' + self.organ + '/' + self.view + '/' + self.transformation + '/' \
                                + sub_version + '/' + data_features_sampled['Sex'].map(inv_dict_sexes_to_values) + '/'
            file_names = 'imagetypeplaceholder_' + self.target + '_' + self.organ + '_' + self.view + '_' + \
                         self.transformation + '_' + sub_version + '_' + data_features_sampled['Sex'].map(inv_dict_sexes_to_values)
            if self.leftright:
                activations_path += '/sideplaceholder'
                file_names += '_sideplaceholder'
            data_features_sampled['save_title'] = activations_path + file_names

            self.d_data_features[target] = data_features_sampled
    
    def preprocessing(self):
        print('preprocessing')
        self._select_best_models()
        self._select_outer_folds()
        self._select_eids()
        self._select_samples()
    
    def _preprocess_for_outer_fold(self, outer_fold):
        print('_preprocess_for_outer_fold')
        self.data_features_outer_fold = self.data_features[self.data_features['outer_fold'] == outer_fold]
        self.n_images = len(self.data_features_outer_fold.index)

        if self.leftright:
            self.n_images *= 2
        
        # Generate the data generator(s)
        self.n_images_batch = self.n_images // self.batch_size * self.batch_size
        self.n_samples_batch = self.n_images_batch // 2 if self.leftright else self.n_images_batch
        self.df_batch = self.data_features_outer_fold.iloc[:self.n_samples_batch, :]
        self.df_batch.set_index('eid', inplace=True)
        if self.n_images_batch > 0:
            self.generator_batch = \
                MyImageDataGenerator(organ=self.organ, view=self.view,
                                     data_features=self.df_batch, n_samples_per_subepoch=None,
                                     batch_size=self.batch_size, training_mode=False,
                                     side_predictors=self.side_predictors, dir_images=self.dir_images,
                                     images_width=self.image_width, images_height=self.image_height,
                                     data_augmentation=False, data_augmentation_factor=None, seed=self.seed)            
            
        else:
            self.generator_batch = None
        self.n_samples_leftovers = self.n_images % self.batch_size
        self.df_leftovers = self.data_features_outer_fold.iloc[self.n_samples_batch:, :]
        self.df_leftovers.set_index('eid', drop=False, inplace=True)
        if self.n_samples_leftovers > 0:
            print('left overs:', self.n_samples_leftovers)
            self.generator_leftovers = \
                MyImageDataGenerator(organ=self.organ, view=self.view,
                                     data_features=self.df_leftovers, n_samples_per_subepoch=None,
                                     batch_size=self.n_samples_leftovers, training_mode=False,
                                     side_predictors=self.side_predictors, dir_images=self.dir_images,
                                     images_width=self.image_width, images_height=self.image_height,
                                     data_augmentation=False, data_augmentation_factor=None, seed=self.seed)
        else:
            self.generator_leftovers = None
        
        # load the weights for the fold (for test images in fold i, load the corresponding model: (i-1)%N_CV_folds
        outer_fold_model = str((int(outer_fold) - 1) % self.n_CV_outer_folds)
        self.model.load_weights(self.spath + 'weights/model-weights_' + self.version + '_' + outer_fold_model + '.h5')
        
    @staticmethod
    @tf.custom_gradient
    def guidedRelu(x):
        def grad(dy):
            return tf.cast(dy>0,tf.float32)  * tf.cast(x>0,tf.float32) * dy
        return tf.nn.relu(x), grad
    
    @staticmethod
    def _process_saliency(saliencies):
        print('_process_saliency')
        saliency_batch = []
        for saliency_map in saliencies:
            r_ch = saliency_map.copy()
            r_ch[r_ch < 0] = 0
            b_ch = -saliency_map.copy()
            b_ch[b_ch < 0] = 0
            g_ch = saliency_map.copy() * 0
            a_ch = np.maximum(b_ch, r_ch)
            saliency_map = np.dstack((r_ch, g_ch, b_ch, a_ch))
            saliency_batch.append(saliency_map)
        saliency_batch = np.array(saliency_batch)
            
        return saliency_batch
    
    @staticmethod
    def _standardize_array(array, color=False):
        if not color :
            array_stand = ((array/np.max(array))*255).astype(int)
        else:
            a0 = ((array[:,:,0]/np.max(array[:,:,0]))*255).astype(int)
            a1 = ((array[:,:,1]/np.max(array[:,:,1]))*255).astype(int)
            a2 = ((array[:,:,2]/np.max(array[:,:,2]))*255).astype(int)
            array_stand = np.dstack((a0, a1, a2))
        return array_stand

    
    def _create_saliency(self, Xs, y):
        print('_create_saliency')
        X0 = Xs[0]
        X1 = Xs[1]
        model = self.model
        with tf.GradientTape() as tape:
            tape.watch(X0)
            y_pred = model(Xs)
            loss = model.loss(y, y_pred)
        gradients = tape.gradient(loss, X0)
        saliencies = tf.reduce_sum(tf.abs(gradients), axis=-1).numpy()
        for i in range(saliencies.shape[0]):
            saliencies[i] =  saliencies[i] * (255 / np.max(np.abs(saliencies[i])))
        saliencies = saliencies.astype(int)
        return saliencies
    
    def _create_smooth_saliency(self, Xs, y, stdev_spread=0.1, n=5, magnitude=False):
        print('_create_smooth_saliency')
        image = Xs[0]
        side_predictors = Xs[1]
        model = self.model
        stdev = stdev_spread * (np.max(image.numpy(), axis=(1,2,3)) - np.min(image.numpy(), axis=(1,2,3)))
        stdev = stdev.reshape(image.shape[0], 1, 1, 1)
        total_gradients = np.zeros_like(image[:,:,:,1])

        for i in range(n):
            noise = np.random.normal(loc=0, scale = stdev, size=image.shape)
            image_noise = image + noise
            with tf.GradientTape() as tape:
                tape.watch(image_noise)
                preds = self.model([image_noise, side_predictors])
            grads = tape.gradient(preds, image_noise)
            grads = tf.reduce_sum(tf.abs(grads), axis=-1).numpy()
            if magnitude:
                grads *= grads
            total_gradients += grads
        saliencies = total_gradients / n
        
        for i in range(saliencies.shape[0]):
            saliencies[i] =  saliencies[i] * (255 / np.max(np.abs(saliencies[i])))
        saliencies = saliencies.astype(int)
        return saliencies
    
    def _create_gradcam(self, Xs, y):
        print('_create_gradcam')
        model = self.model
        l_gradcams = []
        l_raw_heatmaps = []
        l_raw_heatmaps_1D = []

        for idx_img in range(Xs[0].shape[0]):
            img = Xs[0][idx_img][np.newaxis, ...]
            sp = Xs[1][idx_img][np.newaxis, ...]
            # creating model to get feature maps of last CNN layer
            grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(self.last_conv_layer).output, model.output])
            # compute gradient in respect of the feature maps
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model([img, sp])
            grads = tape.gradient(predictions, conv_output)
            # global average pooling to get one weight per feature map
            weights = tf.reduce_mean(tf.abs(grads), axis=(0, 1, 2))
            # compute the weighted combination of feature maps
            heatmap = np.zeros((conv_output.shape[1], conv_output.shape[2]), dtype=np.float32)
            for i, w in enumerate(weights):
                heatmap += w * conv_output[0, :, :, i]
            # reshape and relu the results
            heatmap = cv2.resize(heatmap.numpy(), (img.shape[2], img.shape[1]))
            heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
            l_raw_heatmaps_1D.append(self._standardize_array(heatmap))
            # add colors
            heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            raw_heatmap = heatmap.copy()
            # normalization required for next step (inputs between 0 and 1)
            heatmap = np.float32(heatmap) / 255
            # blending image with original one
            heatmap = cv2.addWeighted(heatmap, 0.5, np.float32(img[0].numpy()), 0.5, 0)
            gradcam = np.uint8(255 * heatmap)
            l_gradcams.append(gradcam)
            l_raw_heatmaps.append(raw_heatmap)
        # reformating in a array
        l_gradcams = np.stack(l_gradcams, axis=0)
        l_raw_heatmaps = np.stack(l_raw_heatmaps, axis=0)
        l_raw_heatmaps_1D = np.stack(l_raw_heatmaps_1D, axis=0)
        
        return l_gradcams, l_raw_heatmaps_1D, l_raw_heatmaps
    
    def _create_guided_backpropagation(self, Xs, y):
        print('_create_guided_backpropagation')
        model = self.model
        l_guided_backpropgations = []
        for idx_img in range(Xs[0].shape[0]):
            # getting data under the dimensions [1, width, height, ?channel]
            img = Xs[0][idx_img][np.newaxis, ...]
            sp = Xs[1][idx_img][np.newaxis, ...]
            # creating model to get feature maps of last CNN layer
            layer_dict = [layer for layer in model.layers[1:] if hasattr(layer,'activation')]
            for layer in layer_dict:
                if layer.activation == tf.keras.activations.relu:
                    layer.activation = self.guidedRelu
            with tf.GradientTape() as tape:
                tape.watch(img)
                prediction = model([img, sp])[0][0]
            grads = tape.gradient(prediction, img)
            # taking into account negative gradients
            grads = np.abs(grads)
            # scaling up
            guided_backpropgation = grads / np.max(grads)
            # guided_backpropgation = guided_backpropgation.sum(axis=3)
            guided_backpropgation = np.uint8(255 * guided_backpropgation).round().astype(int)
            l_guided_backpropgations.append(guided_backpropgation.squeeze())
            
        # reformating in a array
        l_guided_backpropgations = np.stack(l_guided_backpropgations, axis=0)
        return l_guided_backpropgations
    
    def _create_guided_gradcam(self, RawGradcams1D, RawGradcams, GuidedBackpropagations):
        print('_create_guided_gradcam')
        
        # making sure array multriplication works (no uint8)
        RawGradcams = RawGradcams.astype(int)
        RawGradcams1D = RawGradcams1D.astype(int)
        GuidedBackpropagations = GuidedBackpropagations.astype(int)
        
        # creating list of arrays
        l_guided_gradcams_1D = []
        l_guided_gradcams = []
        l_guided_gradcams_test = []

        # creating guided gradcams for 1D and 3D
        for idx_img in range(RawGradcams.shape[0]):
            # standarizing gradcam
            raw_gradcam_1D = self._standardize_array(RawGradcams1D[idx_img].squeeze())
            gradcam = self._standardize_array(RawGradcams[idx_img].squeeze(), color=True)
            gradcam_test = self._standardize_array(RawGradcams[idx_img].squeeze())
            
            # standardizing guided backpropagation
            guided_backpropagation_1D  = self._standardize_array(GuidedBackpropagations[idx_img].squeeze().sum(axis=2))
            guided_backpropagation = self._standardize_array(GuidedBackpropagations[idx_img].squeeze(), color=True)
            guided_backpropagation_test = self._standardize_array(GuidedBackpropagations[idx_img].squeeze())
            
            # creating guided gradcam
            guided_gradcam_1D = self._standardize_array(raw_gradcam_1D*guided_backpropagation_1D)
            guided_gradcam = self._standardize_array(gradcam*guided_backpropagation, color=True)
            guided_gradcam_test = self._standardize_array(gradcam_test*guided_backpropagation_test)
            
            # adding guided gradcam to list of guided gradcams
            l_guided_gradcams_1D.append(guided_gradcam_1D)
            l_guided_gradcams.append(guided_gradcam)
            l_guided_gradcams_test.append(guided_gradcam_test)
        
        # reforming in a array
        a_guided_gradcams_1D = np.stack(l_guided_gradcams_1D, axis=0)
        a_guided_gradcams = np.stack(l_guided_gradcams, axis=0)
        a_guided_gradcams_test = np.stack(l_guided_gradcams_test, axis=0)
        
        return a_guided_gradcams_1D, a_guided_gradcams, a_guided_gradcams_test
        
        
    def _generate_maps_for_one_batch(self, df, Xs, y):
        print('_generate_maps_for_one_batch')
        #create tensors
        X0 = tf.convert_to_tensor(Xs[0])
        X1 = tf.convert_to_tensor(Xs[1])
        Xs = [X0, X1]
        
        #create interpretation maps
        if not self.only_guided_gradcam:
            saliencies = self._create_saliency(Xs, y)
            saliencies = self._process_saliency(saliencies)
            saliencies_smooth = self._create_smooth_saliency(Xs, y, n=15, stdev_spread=0.1)
            
        gradcams, raw_gradcams_1D, raw_gradcams =  self._create_gradcam(Xs, y)
        guided_backpropagations = self._create_guided_backpropagation(Xs, y)
        guided_gradcams_1D, guided_gradcams, guided_gradcams_test = self._create_guided_gradcam(raw_gradcams_1D, raw_gradcams, guided_backpropagations)
                
        
        # Save single images and filters
        for j in range(len(y)):
            # select sample
            if self.leftright:
                idx = j // 2
                side = 'right' if j % 2 == 0 else 'left'
            else:
                idx = j
                side = None
            path = df['save_title'].values[idx]
            ID = df['eid'].values[idx]
            path = path + '_' + str(ID)
            # create directory tree if necessary
            if self.leftright:
                path = path.replace('sideplaceholder', side)
            path_dir = '/'.join(path.split('/')[:-1])
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            # Save raw image
            # Compute path to test if images existed in first place
            path_image = self.ipath + '../images/' + self.organ + '/' + self.view + '/' + self.transformation + '/'
            if self.leftright:
                path_image += side + '/'
            path_image_2 = str(ID) + '_2.jpg'
            path_image_3 = str(ID) + '_3.jpg'
            if os.path.exists(path_image + path_image_2):
                path_image = path_image + path_image_2
            elif os.path.exists(path_image + path_image_3):
                path_image = path_image + path_image_3
            if path_image[-3:] != 'jpg':
                print(f'No image found at {path_image} for ID: {ID}, skipping.')
                continue
            img = load_img(path_image, target_size=self.dict_organ_view_transformation_to_image_size[self.organ + '_' + self.view + '_' + self.transformation])
            img.save(path.replace('imagetypeplaceholder', 'RawImage') + '.jpg')
            
            if not self.only_guided_gradcam:
                # saving saliency
                saliency = saliencies[j, :, :]
                np.save(path.replace('imagetypeplaceholder', 'Saliency') + '.npy', saliency)
                # saving saliency smooth
                saliency_smooth = saliencies_smooth[j, :, :]
                np.save(path.replace('imagetypeplaceholder', 'SaliencySmooth') + '.npy', saliency_smooth)
                # saving gradcam
                gradcam = gradcams[j, :, :]
                np.save(path.replace('imagetypeplaceholder', 'Gradcam') + '.npy', gradcam)
                # saving raw gradcams
                raw_gradcam = raw_gradcams[j, :, :]
                np.save(path.replace('imagetypeplaceholder', 'RawGradcam') + '.npy', raw_gradcam)
                # saving backPropagation
                guided_back_propagation = guided_backpropagations[j, :, :]
                np.save(path.replace('imagetypeplaceholder', 'GuidedBackpropagation') + '.npy', guided_back_propagation)
                # saving gradcam
                guided_gradcam = guided_gradcams[j, :, :]
                np.save(path.replace('imagetypeplaceholder', 'GuidedGradcam') + '.npy', guided_gradcam)  
                # saving gradcam test
                guided_gradcam_test = guided_gradcams_test[j, :, :]
                np.save(path.replace('imagetypeplaceholder', 'GuidedGradcamTest') + '.npy', guided_gradcam_test)  
            # saving gradcam 1D
            guided_gradcam_1D = guided_gradcams_1D[j, :, :]
            np.save(path.replace('imagetypeplaceholder', 'GuidedGradcam1D') + '.npy', guided_gradcam_1D)
            
#             # Define the image types
#             image_types = ['Saliency', 'SaliencySmooth', 'Gradcam', 'RawGradcam', 'GuidedBackpropagation',
#                        'GuidedGradcam1D', 'GuidedGradcam', 'GuidedGradcamTest']
#             # image_types = ['RawGradcam', 'GuidedBackpropagation',
#             #            'GuidedGradcam1D', 'GuidedGradcam', 'GuidedGradcamTest']
#             fig, axes = plt.subplots(1, len(image_types), figsize=(12, 4))


#             for i, image_type in enumerate(image_types):
#                 # Get the corresponding image
#                 image = None
#                 if image_type == 'Saliency':
#                     image = saliencies[j, :, :]
#                 elif image_type == 'SaliencySmooth':
#                     image = saliencies_smooth[j, :, :]
#                 elif image_type == 'Gradcam':
#                     image = gradcams[j, :, :]
#                 elif image_type == 'RawGradcam':
#                     image = raw_gradcams[j, :, :]
#                 elif image_type == 'GuidedBackpropagation':
#                     image = guided_backpropagations[j, :, :]
#                 elif image_type == 'GuidedGradcam1D':
#                     image = guided_gradcams_1D[j, :, :]
#                 elif image_type == 'GuidedGradcam':
#                     image = guided_gradcams[j, :, :]
#                 elif image_type == 'GuidedGradcamTest':
#                     image = guided_gradcams_test[j, :, :]

#                  # Plot the image
#                 axes[i].imshow(image)
#                 axes[i].set_title(image_type)
#                 axes[i].axis('off')

#             # Adjust the layout and display the plot
#             plt.tight_layout()
#             plt.show()

    def generate_filters(self):
        print('generate_filters')
        print('opti:', self.optimizer)
        if len(self.l_outer_folds) == 0:
            print('No outer_fold in common!')
        else:
            for target in self.l_targets:
                print(f'TARGET: {target}')
                self.target = target
                self.version = self.d_versions[target]
                self.parameters = self.d_parameters[target]
                self.data_features = self.d_data_features[target]
                if self.organ + '_' + self.view + '_' + self.transformation in self.organs_views_transformations_images:
                    self.organ = self.parameters['organ']
                    self.view = self.parameters['view']
                    self.transformation = self.parameters['transformation']
                    self.architecture = self.parameters['architecture']
                    self.n_fc_layers = int(self.parameters['n_fc_layers'])
                    self.n_fc_nodes = int(self.parameters['n_fc_nodes'])
                    self.optimizer = self.parameters['optimizer']
                    self.learning_rate = float(self.parameters['learning_rate'])
                    self.weight_decay = float(self.parameters['weight_decay'])
                    self.dropout_rate = float(self.parameters['dropout_rate'])
                    self.data_augmentation_factor = float(self.parameters['data_augmentation_factor'])
                    self._generate_architecture()
                    self.model.compile(optimizer=self.optimizers[self.optimizer](learning_rate=self.learning_rate, clipnorm=1.0),
                                       loss=self.loss_function, metrics=self.metrics)
                    self.last_conv_layer = self.dict_architecture_to_last_conv_layer_name[self.parameters['architecture']]
                    for outer_fold in self.l_outer_folds:
                        print('Generate attention maps for outer_fold ' + outer_fold)
                        gc.collect()
                        self._preprocess_for_outer_fold(int(outer_fold))
                        n_samples_per_batch = self.batch_size // 2 if self.leftright else self.batch_size
                        for i in range(self.n_images // self.batch_size):
                            print('Generating maps for batch ' + str(i))
                            Xs, y = self.generator_batch.__getitem__(i)
                            df = self.df_batch.iloc[n_samples_per_batch * i: n_samples_per_batch * (i + 1), :].reset_index()
                            self._generate_maps_for_one_batch(df, Xs, y)
                        if self.n_samples_leftovers > 0:
                            print('Generating maps for leftovers')
                            Xs, y = self.generator_leftovers.__getitem__(0)
                            self._generate_maps_for_one_batch(self.df_leftovers, Xs, y)
                            
                            
# This class was coded by Samuel Diai.
class InnerCV:
    
    """
    Helper class to perform an inner cross validation to tune the hyperparameters of models trained on scalar predictors
    """
    
    def __init__(self, models, inner_splits, n_iter):
        self.inner_splits = inner_splits
        self.n_iter = n_iter
        if isinstance(models, str):
            models = [models]
        self.models = models
    
    @staticmethod
    def get_model(model_name, params):
        if model_name == 'ElasticNet':
            return ElasticNet(max_iter=2000, **params)
        elif model_name == 'RandomForest':
            return RandomForestRegressor(**params)
        elif model_name == 'GradientBoosting':
            return GradientBoostingRegressor(**params)
        elif model_name == 'Xgboost':
            return XGBRegressor(**params)
        elif model_name == 'LightGbm':
            return LGBMRegressor(**params)
        elif model_name == 'NeuralNetwork':
            return MLPRegressor(solver='adam',
                                activation='relu',
                                hidden_layer_sizes=(128, 64, 32),
                                batch_size=1000,
                                early_stopping=True, **params)
    
    @staticmethod
    def get_hyper_distribution(model_name):
        
        if model_name == 'ElasticNet':
            return {
                'alpha': hp.loguniform('alpha', low=np.log(0.01), high=np.log(10)),
                'l1_ratio': hp.uniform('l1_ratio', low=0.01, high=0.99)
            }
        elif model_name == 'RandomForest':
            return {
                'n_estimators': hp.randint('n_estimators', upper=300) + 150,
                'max_features': hp.choice('max_features', ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
                'max_depth': hp.choice('max_depth', [None, 10, 8, 6])
            }
        elif model_name == 'GradientBoosting':
            return {
                'n_estimators': hp.randint('n_estimators', upper=300) + 150,
                'max_features': hp.choice('max_features', ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
                'learning_rate': hp.uniform('learning_rate', low=0.01, high=0.3),
                'max_depth': hp.randint('max_depth', 10) + 5
            }
        elif model_name == 'Xgboost':
            return {
                'colsample_bytree': hp.uniform('colsample_bytree', low=0.2, high=0.7),
                'gamma': hp.uniform('gamma', low=0.1, high=0.5),
                'learning_rate': hp.uniform('learning_rate', low=0.02, high=0.2),
                'max_depth': hp.randint('max_depth', 10) + 5,
                'n_estimators': hp.randint('n_estimators', 300) + 150,
                'subsample': hp.uniform('subsample', 0.2, 0.8)
            }
        elif model_name == 'LightGbm':
            return {
                'num_leaves': hp.randint('num_leaves', 40) + 5,
                'min_child_samples': hp.randint('min_child_samples', 400) + 100,
                'min_child_weight': hp.choice('min_child_weight', [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]),
                'subsample': hp.uniform('subsample', low=0.2, high=0.8),
                'colsample_bytree': hp.uniform('colsample_bytree', low=0.4, high=0.6),
                'reg_alpha': hp.choice('reg_alpha', [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]),
                'reg_lambda': hp.choice('reg_lambda', [0, 1e-1, 1, 5, 10, 20, 50, 100]),
                'n_estimators': hp.randint('n_estimators', 300) + 150
            }
        elif model_name == 'NeuralNetwork':
            return {
                'learning_rate_init': hp.loguniform('learning_rate_init', low=np.log(5e-5), high=np.log(2e-2)),
                'alpha': hp.uniform('alpha', low=1e-6, high=1e3)
            }
    
    def create_folds(self, X, y):
        """
        X columns : eid + features except target
        y columns : eid + target
        """
        X_eid = X.drop_duplicates('eid')
        y_eid = y.drop_duplicates('eid')
        eids = X_eid.reset_index(drop=True).eid

        
        # Kfold on the eid, then regroup all ids
        # two can be in the same ?
        inner_cv = KFold(n_splits=self.inner_splits, shuffle=True, random_state=0)
        # array for test/val part (separated into 10 ?)
        list_test_folds = [elem[1] for elem in inner_cv.split(X_eid, y_eid)]
        # eids for test/val
        list_test_folds_eid = [eids[elem].values for elem in list_test_folds]
        # not eids but ids for dataframes
        return list_test_folds_eid
    
    def optimize_hyperparameters(self, X, y, scoring):
        """
        input X  : dataframe with features + eid
        input y : dataframe with target + eid
        """
        list_test_folds_id = self.create_folds(X, y)
        X = X.drop(columns=['eid'])
        y = y.drop(columns=['eid'])
        
        # Create custom Splits
        # list new array per fold
        list_test_folds_id_index = [np.array([X.index.get_loc(elem) for elem in list_test_folds_id[fold_num]])
                                    for fold_num in range(len(list_test_folds_id))]
        # create test_folds where we put the number of the fold of test where the values of X are supposed to be
        test_folds = np.zeros(len(X), dtype='int')
        for fold_count in range(len(list_test_folds_id)):
            test_folds[list_test_folds_id_index[fold_count]] = fold_count
        # split using the custom just done before
        inner_cv = PredefinedSplit(test_fold=test_folds)
        
        list_best_params = {}
        list_best_score = {}
        objective, model_name = None, None
        for model_name in self.models:
            def objective(hyperparameters):
                # get elastic serach 
                estimator_ = self.get_model(model_name, hyperparameters)
                # pipeline: standardize -> estimate
                pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', estimator_)])
                # train la regression sur train et test sur test ?
                scores = cross_validate(pipeline, X.values, y, scoring=scoring, cv=inner_cv, n_jobs=self.inner_splits)
                # 'STATUS_OK' typically represents a successful optimization run.
                return {'status': STATUS_OK, 'loss': -scores['test_score'].mean(),
                        'attachments': {'split_test_scores_and_params': (scores['test_score'], hyperparameters)}}
            # get hyperparmameters
            space = self.get_hyper_distribution(model_name)
            # creates an instance of the Trials class to track the results of hyperparameter optimization.
            trials = Trials()
            #objective: The objective function to minimize
            #space: The search space for hyperparameters
            #algo=tpe.suggest: The algorithm to use for optimization. tpe.suggest stands for Tree-structured Parzen Estimator, which is a Bayesian optimization algorithm 
            #The maximum number of iterations for hyperparameter search
            #fmin function performs the optimization by sampling different sets of hyperparameters from the search space, evaluating them using the objective function, and updating the trials object with the results of each trial. It returns the hyperparameter configuration that resulted in the best performance
            best = fmin(objective, space, algo=tpe.suggest, max_evals=self.n_iter, trials=trials)
            #best_params = space_eval(space, best) retrieves the best set of hyperparameters by evaluating the best configuration obtained from fmin in the original search space 
            best_params = space_eval(space, best)
            list_best_params[model_name] = best_params
            # stores the best score obtained during hyperparameter optimization
            list_best_score[model_name] = - min(trials.losses())
        
        # Recover best between all models :
        best_model = max(list_best_score.keys(), key=(lambda k: list_best_score[k]))
        best_model_hyp = list_best_params[best_model]
        
        # Recreate best estim :
        estim = self.get_model(best_model, best_model_hyp)
        pipeline_best = Pipeline([('scaler', StandardScaler()), ('estimator', estim)])
        pipeline_best.fit(X.values, y)
        return pipeline_best
                    
                    
                    
"""
Useful for EnsemblesPredictions. This function needs to be global to allow pool to pickle it.
"""
def compute_ensemble_folds(ensemble_inputs):
    if len(ensemble_inputs[1]) < 100:
        print('Small sample size:' + str(len(ensemble_inputs[1])))
        n_inner_splits = 5
    else:
        n_inner_splits = 10
    # Can use different models: models=['ElasticNet', 'LightGBM', 'NeuralNetwork']
    # regression model that combines both L1 (Lasso) and L2 (Ridge) regularization techniques
    cv = InnerCV(models=['ElasticNet'], inner_splits=n_inner_splits, n_iter=30)
    model = cv.optimize_hyperparameters(ensemble_inputs[0], ensemble_inputs[1], scoring='r2')
    return model

class PerformancesTuning(Metrics):
    
    """
    For each model, selects the best hyperparameter combination.
    """
    
    def __init__(self, target=None):
        
        Metrics.__init__(self)
        self.target = target
        self.PERFORMANCES = {}
        self.PREDICTIONS = {}
        self.Performances = None
        self.models = None
        self.folds = ['val', 'test']
    
    def load_data(self):
        for fold in self.folds:
            path = self.spath + 'dataframes/PERFORMANCES_withoutEnsembles_ranked_' + self.target + \
                   '_'  + fold + '.csv'
            self.PERFORMANCES[fold] = pd.read_csv(path).set_index('version', drop=False)
            self.PERFORMANCES[fold]['organ'] = self.PERFORMANCES[fold]['organ'].astype(str)
            self.PERFORMANCES[fold].index.name = 'columns_names'
            self.PREDICTIONS[fold] = pd.read_csv(path.replace('PERFORMANCES', 'PREDICTIONS').replace('_ranked', ''))
    
    def preprocess_data(self):
        # Get list of distinct models without taking into account hyperparameters tuning
        self.Performances = self.PERFORMANCES['val']
        self.Performances['model'] = self.Performances['organ'] + '_' + self.Performances['view'] + '_' + \
                                     self.Performances['transformation'] + '_' + self.Performances['architecture']
        self.models = self.Performances['model'].unique()
    
    def select_models(self):
        main_metric_name = self.dict_main_metrics_names[self.project] #RMSE
        main_metric_mode = self.main_metrics_modes[main_metric_name]
        Perf_col_name = main_metric_name + '_all'
        for model in self.models:
            Performances_model = self.Performances[self.Performances['model'] == model]
            Performances_model.sort_values([Perf_col_name, 'n_fc_layers', 'n_fc_nodes', 'learning_rate', 'dropout_rate',
                                            'weight_decay', 'data_augmentation_factor'],
                                           ascending=[main_metric_mode == 'min', True, True, False, False, False,
                                                      False], inplace=True)
            best_version = Performances_model['version'].iloc[0]
            # print(best_version)
            # print(f'best_version:{best_version}')
            versions_to_drop = [version for version in Performances_model['version'].values if
                                not version == best_version]
            # define columns from predictions to drop
            cols_to_drop = ['pred_' + version for version in versions_to_drop]
            for fold in self.folds:
                # print(self.PREDICTIONS[fold].columns)
                self.PERFORMANCES[fold].drop(versions_to_drop, inplace=True)
                self.PREDICTIONS[fold].drop(cols_to_drop, axis=1, inplace=True)
        
        # drop 'model' column
        self.Performances.drop(['model'], axis=1, inplace=True)
        
        # Display results
        for fold in self.folds:
            print('The tuned ' + fold + ' performances are:')
            # print(self.PERFORMANCES[fold])
    
    def save_data(self):
        # Save the files
        for fold in self.folds:
            path_pred = self.spath + 'dataframes/PREDICTIONS_tuned_' + self.target + '_' +fold + \
                        '.csv'
            path_perf = self.spath + 'dataframes/PERFORMANCES_tuned_ranked_' + self.target + '_' +\
                        fold + '.csv'
            self.PREDICTIONS[fold].to_csv(path_pred, index=False)
            self.PERFORMANCES[fold].to_csv(path_perf, index=False)
            Performances_alphabetical = self.PERFORMANCES[fold].sort_values(by='version')
            Performances_alphabetical.to_csv(path_perf.replace('ranked', 'alphabetical'), index=False)

            
            
class EnsemblesPredictions(Metrics):
    
    """
    Hierarchically builds ensemble models from the already existing predictions.
    """
    
    def __init__(self, target=None, regenerate_models=False):
        # Parameters
        Metrics.__init__(self)
        
        self.target = target
        self.folds = ['val', 'test']
        self.regenerate_models = regenerate_models
        self.ensembles_performance_cutoff_percent = 0.5 # what is it ?
        self.parameters = {'target': self.target, 'organ': '*', 'view': '*', 'transformation': '*', 'architecture': '*',
                           'n_fc_layers': '*', 'n_fc_nodes': '*', 'optimizer': '*', 'learning_rate': '*',
                           'weight_decay': '*', 'dropout_rate': '*', 'data_augmentation_factor': '*'}
        self.version = self._parameters_to_version(self.parameters)
        
        self.main_metric_name = self.dict_main_metrics_names[self.project]
        self.init_perf = -np.Inf if self.main_metrics_modes[self.main_metric_name] == 'max' else np.Inf
        path_perf = self.spath + 'dataframes/PERFORMANCES_tuned_ranked_' + self.target + '_val.csv'
        self.Performances = pd.read_csv(path_perf).set_index('version', drop=False)
        self.Performances['organ'] = self.Performances['organ'].astype(str)
        self.list_ensemble_levels = ['transformation', 'view', 'organ']
        self.PREDICTIONS = {}
        self.weights_by_category = None
        self.weights_by_ensembles = None
        
        self.N_ensemble_CV_split = len(self.outer_folds)

    # Get rid of columns and rows for the versions for which all samples as NANs
    @staticmethod
    def _drop_na_pred_versions(PREDS, Performances):
        # Select the versions for which only NAs are available
        pred_versions = [col for col in PREDS['val'].columns.values if 'pred_' in col]
        to_drop = []
        for pv in pred_versions:
            for fold in PREDS.keys():
                if PREDS[fold][pv].notna().sum() == 0:
                    to_drop.append(pv)
                    break
        
        # Drop the corresponding columns from preds, and rows from performances
        index_to_drop = [p.replace('pred_', '') for p in to_drop if '*' not in p]
        for fold in PREDS.keys():
            PREDS[fold].drop(to_drop, axis=1, inplace=True)
        return Performances.drop(index_to_drop)
    
    def load_data(self):
        ('    load_data')
        for fold in self.folds:
            predictions_folds = pd.read_csv(self.spath + 'dataframes/PREDICTIONS_tuned_' + self.target + '_' + fold + '.csv')
            l_cols_pred = [col for col in predictions_folds.columns if 'pred' in col]
            for col in l_cols_pred:
                if predictions_folds[col].isna().any():
                    predictions_folds = predictions_folds.drop([col], axis=1)
            
            self.PREDICTIONS[fold] = predictions_folds
        
    
    def _build_single_ensemble(self, PREDICTIONS, version):
        print('    _build_single_ensemble')
        # Drop columns that are exclusively NaNs
        # if one columns is entirely empty, remove it from val and test 
        all_nan = PREDICTIONS['val'].isna().all() | PREDICTIONS['test'].isna().all()
        non_nan_cols = all_nan[~all_nan.values].index
        for fold in self.folds:
            PREDICTIONS[fold] = PREDICTIONS[fold][non_nan_cols]
        Predictions = PREDICTIONS['val']
        # Select the columns for the model
        # will be True if the string col starts with the pattern 'pred_' + version, and False otherwise. There are * in the version -> compile take them
        ensemble_preds_cols = [col for col in Predictions.columns.values if
                                    bool(re.compile('pred_' + version).match(col))]
        
        # If only one model in the ensemble, just copy the column. Otherwise build the ensemble model
        if len(ensemble_preds_cols) == 1:
            for fold in self.folds:
                PREDICTIONS[fold]['pred_' + version] = PREDICTIONS[fold][ensemble_preds_cols[0]]
        else:
            # Initiate the dictionaries
            PREDICTIONS_OUTERFOLDS = {}
            ENSEMBLE_INPUTS = {}
            for outer_fold in self.outer_folds:
                # take the subset of the rows that correspond to the outer_fold
                PREDICTIONS_OUTERFOLDS[outer_fold] = {}
                XS_outer_fold = {}
                YS_outer_fold = {}
                dict_fold_to_outer_folds = {
                    'val': [float(outer_fold)],
                    'test': [(float(outer_fold) + 1) % self.n_CV_outer_folds],
                    'train': [float(of) for of in self.outer_folds
                              if float(of) not in [float(outer_fold), (float(outer_fold) + 1) % self.n_CV_outer_folds]]
                }
                
                for fold in self.folds:
                    # print(f'PREDICTION - {outer_fold} - {fold} \n', PREDICTIONS[fold][PREDICTIONS[fold]['outer_fold'].isin(dict_fold_to_outer_folds[fold])].head(2))
                    PREDICTIONS_OUTERFOLDS[outer_fold][fold] = \
                        PREDICTIONS[fold][PREDICTIONS[fold]['outer_fold'].isin(dict_fold_to_outer_folds[fold])]
                    PREDICTIONS_OUTERFOLDS[outer_fold][fold] = PREDICTIONS_OUTERFOLDS[outer_fold][fold][
                        ['eid', 'target'] + ensemble_preds_cols].dropna()
                    X = PREDICTIONS_OUTERFOLDS[outer_fold][fold][['eid'] + ensemble_preds_cols]
                    X.set_index('eid', inplace=True, drop=False)
                    XS_outer_fold[fold] = X
                    y = PREDICTIONS_OUTERFOLDS[outer_fold][fold][['eid', 'target']]
                    y.set_index('eid', inplace=True, drop=False)
                    YS_outer_fold[fold] = y
                    # if outer_fold == 0:
                    #     print(XS_outer_fold[fold], YS_outer_fold[fold])
                ENSEMBLE_INPUTS[outer_fold] = [XS_outer_fold['val'], YS_outer_fold['val']]
            # Build ensemble model using ElasticNet and/or LightGBM, Neural Network.
            PREDICTIONS_ENSEMBLE = {}
            pool = Pool(self.N_ensemble_CV_split)
            print('ENSEMBLE_INPUTS.keys :', ENSEMBLE_INPUTS.keys())
            print('inner cv ENSEMBLE INPUTS len: ', len(list(ENSEMBLE_INPUTS.values())))
            global test
            
            MODELS = pool.map(compute_ensemble_folds, list(ENSEMBLE_INPUTS.values()))
            pool.close()
            pool.join()
            test = MODELS
            
            # Concatenate all outer folds
            for outer_fold in self.outer_folds:
                for fold in self.folds:
                    X = PREDICTIONS_OUTERFOLDS[outer_fold][fold][ensemble_preds_cols]
                    PREDICTIONS_OUTERFOLDS[outer_fold][fold]['pred_' + version] = MODELS[int(outer_fold)].predict(X)
                    # PREDICTIONS_OUTERFOLDS[outer_fold][fold]['pred_' + version] = MODELS[0].predict(X)
                    PREDICTIONS_OUTERFOLDS[outer_fold][fold]['outer_fold'] = float(outer_fold)
                    df_outer_fold = PREDICTIONS_OUTERFOLDS[outer_fold][fold][['eid', 'outer_fold',
                                                                              'pred_' + version]]
                    # Initiate, or append if some previous outerfolds have already been concatenated
                    if fold not in PREDICTIONS_ENSEMBLE.keys():
                        PREDICTIONS_ENSEMBLE[fold] = df_outer_fold
                    else:
                        PREDICTIONS_ENSEMBLE[fold] = PREDICTIONS_ENSEMBLE[fold].append(df_outer_fold)
    
            # Add the ensemble predictions to the dataframe
            for fold in self.folds:
                if fold == 'train':
                    PREDICTIONS[fold] = PREDICTIONS[fold].merge(PREDICTIONS_ENSEMBLE[fold], how='outer',
                                                                on=['eid', 'outer_fold'])
                else:
                    PREDICTIONS_ENSEMBLE[fold].drop('outer_fold', axis=1, inplace=True)
                    PREDICTIONS[fold] = PREDICTIONS[fold].merge(PREDICTIONS_ENSEMBLE[fold], how='outer', on=['eid'])
    
    def _build_single_ensemble_wrapper(self, version, ensemble_level):
        print('    _build_single_ensemble_wrapper')
        # 1. version -> organ view transfo * *
        # 2.ensemble level -> transfo
        print('Building the ensemble model ' + version)
        pred_version = 'pred_' + version
        self._build_single_ensemble(self.PREDICTIONS, version)
        
        # build and save a dataset for this specific ensemble model
        for fold in self.folds:
            df_single_ensemble = self.PREDICTIONS[fold][['eid', 'outer_fold', pred_version]]
            df_single_ensemble.rename(columns={pred_version: 'pred'}, inplace=True)
            df_single_ensemble.dropna(inplace=True, subset=['pred'])
            df_single_ensemble.to_csv(self.spath + 'dataframes/Predictions_' + version + '_' + fold + '.csv', index=False)

    
    def _recursive_ensemble_builder(self, Performances_grandparent, parameters_parent, version_parent,list_ensemble_levels_parent):
        print('    _recursive_ensemble_builder')
        # Compute the ensemble models for the children first, so that they can be used for the parent model
        # 1. select versio with * at organ view and transformation 2. specific organ 3.specific organ and view 4 specific organ view tranfo
        Performances_parent = Performances_grandparent[
            Performances_grandparent['version'].isin(
                fnmatch.filter(Performances_grandparent['version'], version_parent))]
        # if the last ensemble level has not been reached, go down one level and create a branch for each child.
        # Otherwise the leaf has been reached
        if len(list_ensemble_levels_parent) > 0:
            list_ensemble_levels_child = list_ensemble_levels_parent.copy()
            # ['transformation', 'view', 'organ'] -> ['transformation', 'view'] -> ['transformation']
            # 1. ensemble level = 'organ' -> 'transformation'
            ensemble_level = list_ensemble_levels_child.pop()
            # list_children = unique organs
            list_children = Performances_parent[ensemble_level].unique()
            for child in list_children:
               #  self.parameters = {'target': self.target, 'organ': '*', 'view': '*', 'transformation': '*', 'architecture': '*',
               # 'n_fc_layers': '*', 'n_fc_nodes': '*', 'optimizer': '*', 'learning_rate': '*',
               # 'weight_decay': '*', 'dropout_rate': '*', 'data_augmentation_factor': '*'}
                parameters_child = parameters_parent.copy()
                parameters_child[ensemble_level] = child
                # version for one orgna specified not view and transformation -> organ and view -> organ view and transformation
                version_child = self._parameters_to_version(parameters_child)
                # recursive call to the function
                self._recursive_ensemble_builder(Performances_parent, parameters_child, version_child,
                                                 list_ensemble_levels_child)
        else:
            ensemble_level = None
        # 1.specific organ view transformation
        # compute the ensemble model for the parent
        # Check if ensemble model has already been computed. If it has, load the predictions. If it has not, compute it.
        # if we don't have to retrain models and that predictions have already been produced
        if not self.regenerate_models and \
                os.path.exists(self.spath + 'dataframes/Predictions_' + version_parent + '_test.csv'):
            print('The model ' + version_parent + ' has already been computed. Loading it...')
            # just val and test ?
            for fold in self.folds:
                # this part already done before
                df_single_ensemble = pd.read_csv(self.spath + 'dataframes/Predictions_' + version_parent + '_' + fold + '.csv')
                # 1. pred_organ_view_transfo
                df_single_ensemble.rename(columns={'pred': 'pred_' + version_parent}, inplace=True)
                # Add the ensemble predictions to the dataframe
                if fold == 'train':
                    self.PREDICTIONS[fold] = self.PREDICTIONS[fold].merge(df_single_ensemble, how='outer',
                                                                          on=['eid', 'outer_fold'])
                else:
                    # we know it is testing or validation -> only one outer_fold -> don't need it
                    df_single_ensemble.drop(columns=['outer_fold'], inplace=True)
                    self.PREDICTIONS[fold] = self.PREDICTIONS[fold].merge(df_single_ensemble, how='outer', on=['id'])
        else:
            self._build_single_ensemble_wrapper(version_parent, ensemble_level)
        
        # Print a quick performance estimation
        df_model = self.PREDICTIONS['test'][['target', 'pred_' + version_parent]].dropna()
        print(self.main_metric_name + ': ' + str(r2_score(df_model['target'], df_model['pred_' + version_parent])))
        print('The sample size is ' + str(len(df_model.index)) + '.')
    
    def generate_ensemble_predictions(self):
        print('    generate_ensemble_predictions')
        self._recursive_ensemble_builder(self.Performances, self.parameters, self.version, self.list_ensemble_levels)
        
        # Reorder the columns alphabetically
        for fold in self.folds:
            pred_versions = [col for col in self.PREDICTIONS[fold].columns if 'pred_' in col]
            pred_versions.sort()
            self.PREDICTIONS[fold] = self.PREDICTIONS[fold][self.id_vars + self.demographic_vars + pred_versions]
        
        # Displaying the R2s
        for fold in self.folds:
            versions = [col.replace('pred_', '') for col in self.PREDICTIONS[fold].columns if 'pred_' in col]
            r2s = []
            for version in versions:
                df = self.PREDICTIONS[fold][['target', 'pred_' + version]].dropna()
                r2s.append(r2_score(df['target'], df['pred_' + version]))
            R2S = pd.DataFrame({'version': versions, 'R2': r2s})
            R2S.sort_values(by='R2', ascending=False, inplace=True)
            print(fold + ' R2s for each model: ')
            print(R2S)
    
    def save_predictions(self):
        for fold in self.folds:
            self.PREDICTIONS[fold].to_csv(self.spath + 'dataframes/PREDICTIONS_withEnsembles_' + self.target + '_' + fold + '.csv', index=False)

            
            
class AttentionMapsDifference(DeepLearning):
    
    """
    Computes the attention difference maps (saliency maps and Grad_RAM maps) for all images
    """
    
    def __init__(self, t1=None, t2=None, organ=None, view=None, transformation=None):
        # Partial initialization with placeholders to get access to parameters and functions
        DeepLearning.__init__(self, t1['target'], organ , view, transformation, 'InceptionResNetV2', '1', '1024', 'Adam',
                              '0.0001', '0.1', '0.5', '1.0', False)
        # Parameters
        
        self.target1 = t1['target']
        self.target2 = t2['target']
        self.params1 = t1['sub_parameters']
        self.params2 = t2['sub_parameters']
        self.sub_version1 = self._parameters_to_version(self.params1)
        self.sub_version2 = self._parameters_to_version(self.params2)
                
        self.organ = organ
        self.view = view
        self.transformation = transformation
        self.path_figures = '/n/groups/patel/joachim/figures/Attention_Maps/'     
        self.path_maps = self.path_figures + '{target}' + '/' + self.organ + '/' + self.view + '/' + self.transformation + '/' + '{sub_version}' + '/' + '{sex}' + '/'
        self.path_difference_sex = self.path_maps.format(target = 'T1' + '_' + self.target1 + '_' + 'T2' + '_' + self.target2,
                                                         sub_version = 'V1' + '_' + self.sub_version1 + '_' + 'V2' + '_' + self.sub_version2,
                                                         sex = '{sex}')
        self.path_difference = '/'.join(self.path_difference_sex.split('/')[:-2]) + '/'
        self.path_raw_image = '/n/groups/patel/joachim/figures/Attention_Maps/{target}/{organ}/{view}/{transformation}/{sub_version}/{sex}/RawImage_{target}_{organ}_{view}_{transformation}_{sub_version}_{sex}_{eid}.jpg'
        self.path_raw_image = self.path_raw_image.format(target=self.target1, 
                                                         organ=self.organ, 
                                                         view=self.view, 
                                                         transformation=self.transformation, 
                                                         sub_version=self.sub_version1, 
                                                         sex='{sex}',
                                                         eid = '{eid}')
        
        self.d_eids = {}
        
        self.leftright = True if self.organ + '_' + self.view in self.left_right_organs_views else False
        self.dict_sexes_to_values = {'Male': '1.0', 'Female': '0.0'}
        self.l_sexes = ['Male', 'Female']
        self.d_fullbody_boundingboxs = \
        { 'Musculoskeletal': 
             {'Male': {
                'right_foot': {'x1':0, 'x2':90, 'y1':500, 'y2':540},
                'left_foot': {'x1':90, 'x2':180, 'y1':500, 'y2':540},
                'right_leg': {'x1':0, 'x2':90, 'y1':375, 'y2':500},
                'left_leg': {'x1':90, 'x2':180, 'y1':375, 'y2':500},
                'right_thigh': {'x1':25, 'x2':90, 'y1':260, 'y2':375},
                'left_thigh': {'x1':90, 'x2':155, 'y1':260, 'y2':375},
                'lower_chest': {'x1':30, 'x2':150, 'y1':175, 'y2':260},
                'higher_chest': {'x1':30, 'x2':150, 'y1':70, 'y2':175},
                'head': {'x1':30, 'x2':150, 'y1':0, 'y2':70},
                'right_arm': {'x1':0, 'x2':30, 'y1':70, 'y2':175},
                'left_arm': {'x1':150, 'x2':180, 'y1':70, 'y2':175},
                'right_forearm': {'x1':0, 'x2':30, 'y1':175, 'y2':245},
                'left_forearm': {'x1':150, 'x2':180, 'y1':175, 'y2':245},
                'right_hand': {'x1':0, 'x2':25, 'y1':245, 'y2':335},
                'left_hand': {'x1':155, 'x2':180, 'y1':245, 'y2':335}
                },
             'Female':{
                'right_foot': {'x1':10, 'x2':90, 'y1':500, 'y2':540},
                'left_foot': {'x1':90, 'x2':170, 'y1':500, 'y2':540},
                'right_leg': {'x1':25, 'x2':90, 'y1':375, 'y2':500},
                'left_leg': {'x1':90, 'x2':155, 'y1':375, 'y2':500},
                'right_thigh': {'x1':25, 'x2':90, 'y1':260, 'y2':375},
                'left_thigh': {'x1':90, 'x2':155, 'y1':260, 'y2':375},
                'lower_chest': {'x1':28, 'x2':152, 'y1':175, 'y2':260},
                'higher_chest': {'x1':34, 'x2':146, 'y1':70, 'y2':175},
                'head': {'x1':30, 'x2':150, 'y1':0, 'y2':70},
                'right_arm': {'x1':0, 'x2':34, 'y1':70, 'y2':175},
                'left_arm': {'x1':146, 'x2':180, 'y1':70, 'y2':175},
                'right_forearm': {'x1':0, 'x2':28, 'y1':175, 'y2':230},
                'left_forearm': {'x1':152, 'x2':180, 'y1':175, 'y2':230},
                'right_hand': {'x1':0, 'x2':25, 'y1':230, 'y2':325},
                'left_hand': {'x1':155, 'x2':180, 'y1':230, 'y2':325}
             }
            }
        }
        self.colors = {'Musculoskeletal':
                       ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'pink', 'brown', 'gray', 'olive', 'teal', 'navy', 'lime']
                      }
        self.reshape_treshold = 50
        self.reshape_channel = 1
        self.channels={'environmental':0, 'genetic':2}
        self.tolerance = [30,30,30]
    
    def _select_eids_per_sex_per_target(self):
        print('_select_eids_per_sex_per_target')
        # select eids for each target for which attention maps where made
        self.d_eids[self.target1] = {}
        self.d_eids[self.target2] = {}
        
        if not self.leftright:
            for sex in self.l_sexes:
                l_filename_target1_sex = os.listdir(self.path_maps.format(target=self.target1, sub_version=self.sub_version1, sex=sex))
                l_filename_target2_sex = os.listdir(self.path_maps.format(target=self.target2, sub_version=self.sub_version2, sex=sex))
                l_eids_target1_sex = list(set([filename.split('_')[-1].split('.')[0] for filename in l_filename_target1_sex]))
                l_eids_target2_sex = list(set([filename.split('_')[-1].split('.')[0] for filename in l_filename_target2_sex]))
                self.d_eids[self.target1][sex] = l_eids_target1_sex
                self.d_eids[self.target2][sex] = l_eids_target2_sex
        else:
            print('_select_eids_per_sex_per_target is not coded yet for left/right organs!')
            
    def _take_eids_intersection(self):
        print('_take_eids_intersection')
        # process the intersection of eids of the two targets
        self.d_eids['intersection'] = {}
        if not self.leftright:
            for sex in self.l_sexes:
                l_eids_intersection_sex = [eid for eid in self.d_eids[self.target1][sex] if eid in self.d_eids[self.target2][sex]]
                self.d_eids['intersection'][sex] = l_eids_intersection_sex
                print(f'For {sex}, there was {len(l_eids_intersection_sex)} eids in common!')
        else:
            print('_take_eids_intersection is not coded yet for left/right organs!')  
            
    def _compute_saliency_difference(self):
        print('_compute_saliency_difference')
        # create an new image based on both guided gradcam 1D images
        if not self.leftright:
            for sex in self.l_sexes:
                for eid in self.d_eids['intersection'][sex]:
                    path_image_sex = self.path_maps  + 'GuidedGradcam1D_' + '{target}' + '_' + self.organ + '_' + self.view + '_' + self.transformation + '_' \
                     + '{sub_version}' + '_' + sex + '_' + eid + '.npy'
                    # load both images
                    img1 = np.load(path_image_sex.format(target=self.target1, sub_version=self.sub_version1, sex=sex))
                    img2 = np.load(path_image_sex.format(target=self.target2, sub_version=self.sub_version2, sex=sex))
                    # create an new image with each 1channel image on a different channel dimension
                    new_image = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
                    new_image[:, :, 2] = img1
                    new_image[:, :, 0] = img2
                    new_image[:, :, 1] = 0
                    new_image_name = f'GuidedGradcam1DDifference_{eid}.npy'
                    new_path = self.path_difference_sex.format(sex=sex)
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    np.save(new_path + new_image_name, new_image)
        else:
            print('_take_eids_intersection is not coded yet for left/right organs!')  
    @staticmethod
    def _compute_average_absolute_difference(img):
        red_channel = img[:,:,0]
        blue_channel = img[:,:,2]
        absolute_diff = np.abs(blue_channel - red_channel)
        non_black_pixels_mask = (blue_channel != 0) & (red_channel != 0)
        absolute_diff_non_black = absolute_diff[non_black_pixels_mask]
        absolute_mean_pixel_difference = np.mean(absolute_diff_non_black)
        absolute_sum_pixel_difference = np.sum(absolute_diff_non_black)
        return absolute_mean_pixel_difference, absolute_sum_pixel_difference
    
    def _create_bounding_box_whole_body(self, image):
        image_height, image_width = image.shape[:2]
        # get channel to threshold color
        channel = image[:,:,self.reshape_channel]
        # bounding box parameters
        # calculate box if background is yellow
        rows, cols = np.where(channel > self.reshape_treshold)
        x1 = np.min(cols)
        x2 = np.max(cols)
        y1 = np.min(rows)
        y2 = np.max(rows)
        # if background is purple, recalculate box
        if (x1==0) & (y1==0) & (x2==image_width-1) & (y2==image_height-1):
            rows, cols = np.where(channel < self.reshape_treshold)
            x1 = np.min(cols)
            x2 = np.max(cols)
            y1 = np.min(rows)
            y2 = np.max(rows)
        return x1, x2, y1, y2
    
    @staticmethod
    def _create_bounding_box_image(image, bounding_box):
        x1, x2, y1, y2 = bounding_box
        bounding_box_image = image[y1:y2, x1:x2]
        resized_bounding_box_image = cv2.resize(bounding_box_image, (image.shape[1], image.shape[0]))
        return resized_bounding_box_image
    
    @staticmethod
    def _create_body_part_mask(shape, bounding_box):
        height, width = shape
        mask = np.zeros((height, width), dtype=np.float32)
        mask[bounding_box['y1']:bounding_box['y2'], bounding_box['x1']:bounding_box['x2']] = 1
        normalized_mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return normalized_mask
    
    def _create_mask_other(self, shape, sex):
        image_height, image_width = shape
        # Create a blank mask with ones representing the entire image
        mask = np.ones((image_height, image_width), dtype=np.uint8)
        # Iterate over each bounding box and set the corresponding region in the mask to zero
        for box in self.d_fullbody_boundingboxs[self.organ][sex].values():
            x1, x2, y1, y2 = box['x1'], box['x2'], box['y1'], box['y2']
            mask[y1:y2, x1:x2] = 0
        normalized_mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return normalized_mask
    
    def _create_background_mask(self, image):
        # Define the background color using the pixel at position (0,0)
        background_color = image[0, 0]

        # Define lower and upper bounds for the color
        lower = np.array([background_color[0] - self.tolerance[0], background_color[1] - self.tolerance[1], background_color[2] - self.tolerance[2]])
        upper = np.array([background_color[0] + self.tolerance[0], background_color[1] + self.tolerance[1], background_color[2] + self.tolerance[2]])

        # Create a mask for the background
        mask_background = cv2.inRange(image, lower, upper)
        mask_foreground = ~mask_background

        # Save the image
        return mask_background, mask_foreground

    
    def _create_body_part_percentenage(self, image, sex, foreground_mask):
        # calculates body part percentage for each channel 
        d_sum_per_bounding_box = {}
        d_ratio_per_bounding_box = {}
        for channel in list(self.channels.keys()):
            d_sum_per_bounding_box[channel] = {}
            d_ratio_per_bounding_box[channel] = {}
            # get the channel 
            img_channel = image[:,:,self.channels[channel]]
            total_sum = np.sum(img_channel)
            # create list of bounding box / body parts
            l_body_parts = list(self.d_fullbody_boundingboxs[self.organ][sex].keys())
            bounding_boxes = [(body_part, self.d_fullbody_boundingboxs[self.organ][sex][body_part]) for body_part in l_body_parts]
            # for each bounding box, sum the values of the image on the corresponding mask and store it in a dictionnary
            for (body_part, bounding_box) in bounding_boxes:
                mask = self._create_body_part_mask(img_channel.shape[:2], bounding_box)
                mask_without_background = mask*foreground_mask
                sum_region = np.sum(mask/255*img_channel)
                sum_region_without_background = np.sum(mask_without_background/255*img_channel)
                # take the percentage 
                d_sum_per_bounding_box[channel][body_part]=np.round(sum_region/(total_sum), 12)
                d_ratio_per_bounding_box[channel][body_part]=np.round(sum_region_without_background/(total_sum*mask_without_background.sum()), 12)
                
            # adding other
            mask_other = self._create_mask_other(img_channel.shape[:2], sex)
            mask_other_without_background = mask_other*foreground_mask
            sum_region = np.sum(mask_other/255*img_channel)
            sum_region_without_background = np.sum(mask_other_without_background/255*img_channel)

            # avoding NaN
            if mask_other.sum() == 0:
                d_sum_per_bounding_box[channel]['other']=0
            else:
                d_sum_per_bounding_box[channel]['other']=np.round(sum_region/(total_sum), 12)
            if mask_other_without_background.sum() == 0:
                d_ratio_per_bounding_box[channel]['other']=0
            else:
                d_ratio_per_bounding_box[channel]['other']=np.round(sum_region_without_background/(total_sum*mask_other_without_background.sum()), 12)
                
        return d_sum_per_bounding_box, d_ratio_per_bounding_box
    
    def _create_hist_body_part_percentage(self, d_image_body_part_percentage, sex, eid, method):
        
        # if there is only red and blue, continue
        if len(d_image_body_part_percentage.keys()) == 2:
            # get channel names
            red_channel, blue_channel = d_image_body_part_percentage.keys()
            # get order for dataframe
            sorted_body_parts = sorted(
            d_image_body_part_percentage[red_channel].keys(),
            key=lambda x: d_image_body_part_percentage[red_channel][x] + d_image_body_part_percentage[blue_channel][x],
            reverse=True
            )
            # get list of values to plot for each channel 
            red_values = [d_image_body_part_percentage[red_channel][part] for part in sorted_body_parts]
            blue_values = [d_image_body_part_percentage[blue_channel][part] for part in sorted_body_parts]
            # plot the histogram
            plt.figure(figsize=(10, 5))
            bar_width = 0.4
            x = np.arange(len(sorted_body_parts))
            red_bars = plt.bar(x, red_values, width=bar_width, bottom=blue_values, label=red_channel, color='red')
            blue_bars = plt.bar(x, blue_values, width=bar_width, label=blue_channel, color='blue')
            # add labels
            plt.xlabel('Body Parts')
            plt.ylabel('Prediction Understandability Percentage')
            plt.xticks(x, sorted_body_parts, rotation=45)
            plt.legend()
            # save plot
            plt.savefig(self.path_difference_sex.format(sex=sex) + f'/hist_{method}_{eid}.jpg')

        else:
            print('The code for more than 2 channels is not done yet!')

    def _create_patron(self, sex):
        image_shape = self.dict_organ_view_transformation_to_image_size[self.organ + '_' + self.view + '_' + self.transformation]
        black_image = np.zeros(image_shape)
        # Create a figure and axes
        fig, ax = plt.subplots()
        # Plot the image
        ax.imshow(black_image)
        # Plot the bounding box for each body part with a different color
        for i, (body_part, bbox) in enumerate(self.d_fullbody_boundingboxs[self.organ][sex].items()):
            x1, x2, y1, y2 = bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2']
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=self.colors[self.organ][i], facecolor='none')
            ax.add_patch(rect)
        # Add legend outside the image
        legend_handles = []
        for i, (body_part, bbox) in enumerate(self.d_fullbody_boundingboxs[self.organ][sex].items()):
            legend_handles.append(patches.Patch(color=self.colors[self.organ][i], label=body_part))
        # Set the legend position and title
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', title='Body Parts')
        # Remove axis ticks and labels
        ax.axis('off')
        # save the plot
        plt.savefig(self.path_difference +  f'patron_{sex}.jpg')
    
        
    def _create_mean_body_part_percentage(self, df_metrics, method):
        print('_create_mean_body_part_percentage')
        # select columns that will be used
        l_cols = [col for col in df_metrics.columns if ('eid' in col) or (col.split('_')[-1] == method)]
        df_method = df_metrics[l_cols]
        df_method.columns = [col.rstrip(f'_{method}') for col in df_method.columns]

        
        confidence_level = 0.95
        for sex in self.l_sexes:
            l_eids = [int(eid) for eid in self.d_eids['intersection'][sex]] 
            df_method['eid'] = df_method['eid'].astype(int)
            df_sex = df_method.loc[df_method['eid'].isin(l_eids)]
            df_sex = df_sex.drop(columns=['eid'])
            confidence_intervals = pd.DataFrame(columns=df_sex.columns)
            for column in df_sex.columns:
                df_sex[column] = df_sex[column].astype(float)
                if df_sex[column].dtype in [int, float]:
                    data = df_sex[column].dropna()
                    confidence_interval = stats.t.interval(confidence_level, len(data)-1, loc=data.mean(), scale=stats.sem(data))
                    confidence_intervals[column] = confidence_interval
            df_sex_mean = df_sex.mean()
            df_sex_mean = pd.DataFrame({'body_part':df_sex_mean.index, 'values':df_sex_mean.values})
            df_sex_mean[['Category', 'Type']] = df_sex_mean['body_part'].str.rsplit('_', n=1, expand=True)
            df_pivot = df_sex_mean.pivot(index='Category', columns='Type', values='values')
            df_pivot = df_pivot[['environmental', 'genetic']]


            df_pivot['Sum'] = df_pivot.sum(axis=1)
            df_pivot_sorted = df_pivot.sort_values('Sum', ascending=False)

            fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the figure size

            colors = ['red', 'blue']
            labels = ['environmental', 'genetic']

            bar_width = 0.35
            offset = 0.05  # Adjust the offset value

            index = range(len(df_pivot_sorted))

            for i, label in enumerate(labels):
                category_data = df_pivot_sorted[label]
                
                bar_positions = np.array(index) + np.array(i) * bar_width + offset

                ax.bar(bar_positions, category_data, width=bar_width, label=label, color=colors[i], alpha=0.9)

                # Confidence interval data for the category
                ci_data = confidence_intervals.filter(like=label)
                ci_data.columns = ci_data.columns.str.replace('_environmental', '').str.replace('_genetic', '')
                ci_lower = ci_data.iloc[0]
                ci_upper = ci_data.iloc[1]

                errorbar_center = np.array(index) + (np.array(i) * bar_width + offset )

                # Plotting the confidence interval error bars
                ax.errorbar(errorbar_center, category_data, yerr=[category_data - ci_lower, ci_upper - category_data],
                            fmt='none', ecolor='black', capsize=3)
                
            initial_target = self.target1.split('METRIC.')[1].split('.')[0]
            ax.set_xlabel('Body Part')
            ax.set_ylabel(f'Genetic/Environment {initial_target}')
            ax.set_title(f'Histogram of Environmental and Genetic {initial_target} with Confidence Intervals for {sex} with method {method}')
            ax.legend()

            plt.xticks([r + bar_width/2 for r in range(len(df_pivot_sorted))], df_pivot_sorted.index, rotation=45)
            plt.tight_layout()
            plt.savefig(self.path_difference +  f'global_hist_{sex}.jpg')
            
    def _compute_metrics(self, nb_eids):
        print('_compute_metrics')
        # compute metrics
        l_all_eids = self.d_eids['intersection']['Male'] + self.d_eids['intersection']['Female']  
        df_metrics = pd.DataFrame({'eid':l_all_eids, 'absolute_mean_pixel_difference':None})
        # adding body part percentage column:
        for organ_part in self.d_fullbody_boundingboxs[self.organ]['Male'].keys():
            for channel in self.channels:
                df_metrics[f'{organ_part}_{channel}_sum'] = None
                df_metrics[f'{organ_part}_{channel}_ratio'] = None
                
        if not self.leftright:
            for sex in self.l_sexes:
                path_sex = self.path_difference_sex.format(sex=sex)
                for index, eid in enumerate(self.d_eids['intersection'][sex]):
                    if index%10 ==0:
                        print(index)
                    if (nb_eids != None) & (index>=nb_eids):
                        break
                    # load orifinal image and guided gradcam
                    path_img_diff = path_sex + f'GuidedGradcam1DDifference_{eid}.npy'
                    path_raw_img = self.path_raw_image.format(sex=sex, eid=eid)
                    img = np.load(path_img_diff)
                    raw_img = mpimg.imread(path_raw_img)
                    # segment image and project on guided gradcam  
                    _ ,foreground_mask = self._create_background_mask(raw_img)
                    # removing non body parts from guided gradcam
                    img = raw_img*foreground_mask[:,:,np.newaxis]
                    # compute absolute_mean_pixel_difference
                    absolute_mean_pixel_difference, absolute_sum_pixel_difference = self._compute_average_absolute_difference(img)
                    # store the result
                    df_metrics.loc[df_metrics['eid'] == eid, 'absolute_mean_pixel_difference'] = absolute_mean_pixel_difference
                    df_metrics.loc[df_metrics['eid'] == eid, 'absolute_sum_pixel_difference'] = absolute_sum_pixel_difference 
                    # calculate body part percentage
                    d_percentage_per_bounding_box, d_ratio_per_bounding_box = self._create_body_part_percentenage(img, sex, foreground_mask)
                    # store it
                    for channel in d_percentage_per_bounding_box.keys():
                        for body_part in d_percentage_per_bounding_box[channel].keys():
                            df_metrics.loc[df_metrics['eid'] == eid, f'{body_part}_{channel}_sum'] = d_percentage_per_bounding_box[channel][body_part]
                            df_metrics.loc[df_metrics['eid'] == eid, f'{body_part}_{channel}_ratio'] = d_ratio_per_bounding_box[channel][body_part]
                                
                    self._create_hist_body_part_percentage(d_percentage_per_bounding_box, sex, eid, method='sum')
                    self._create_hist_body_part_percentage(d_ratio_per_bounding_box, sex, eid, method='ratio')
                    
                self._create_patron(sex)                
            df_metrics.to_csv(self.path_difference + f'df_metrics.csv')
            df_metrics = df_metrics.dropna()
            self._create_mean_body_part_percentage(df_metrics, 'sum')
            self._create_mean_body_part_percentage(df_metrics, 'ratio')

            
        else:
            print('_take_eids_intersection is not coded yet for left/right organs!')  
    
            
    def preprocessing(self):
        print('preprocessing')
        self._select_eids_per_sex_per_target()
        self._take_eids_intersection()
    def process_difference(self, nb_eids=None):
        print('process_difference')
        self._compute_saliency_difference()
        self._compute_metrics(nb_eids)
    
    
    
    

    
def get_images(path_folder, nb_images, image_size, random=False):
    
    # get list of image paths 
    l_image_paths = glob.glob(os.path.join(path_folder, '*.jpg'))
    # shuffle it
    if random:
        np.random.shuffle(l_image_paths)
    # load and append each image to a list
    l_images = []
    for index, path in enumerate(l_image_paths):
        # stop at when nb_images reached
        if index >= nb_images:
            break
        # load image
        image = mpimg.imread(path)
        # make it into an array and standardize it
        image = Image.fromarray(image)
        image = image.resize(image_size[::-1])
        image = np.array(image)
        image = image.astype('float32') / 255.0  # Normalize pixel values to range [0, 1]
        # add it to the list
        l_images.append(image)
    a_images_std = np.array(l_images)
    return a_images_std
    
    
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class VAE(keras.Model):
    """
    Buiding a (disentangled ?) variational autoencoder.
    """
    
    def __init__(self, input_shape, latent_dim=64, batch_size=64, debug_mode=False, model_name='test'):
        super().__init__()
        # model parameters
        self.latent_dim = latent_dim
        self.input_shape_vae = input_shape
        self.batch_size = 32
        
        # initalizing encoder/decoder
        self.encoder = None
        self.decoder = None
        self.history = None

        # data parameters
        self.path_load_images = '/n/groups/patel/Alan/Aging/Medical_Images/images/Musculoskeletal/FullBody/Mixed/'
        self.path_save_images = '../data/VAE/images/'
        self.path_save_weights = '../data/VAE/weights/'
        self.model_name = model_name + '_' + str(debug_mode)*debug_mode
        self.train_generator = None
        
        # losses 
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        
        # debug mode
        self.debug_mode = debug_mode

        
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def load_data(self, **kwargs):
        # if debug 
        if self.debug_mode == False :
            path_images = kwargs.get('path_images')
            class_name = kwargs.get('class_name')
            train_datagen = ImageDataGenerator(rescale=1./255)
            train_generator = train_datagen.flow_from_directory(
                path_images,
                target_size=self.input_shape_vae[:2],
                batch_size=self.batch_size,
                shuffle=True, 
                classes=[class_name]
            )

            self.train_generator=train_generator
        else :
            nb_images = kwargs.get('nb_images')
            random = kwargs.get('random')
            self.train_generator=get_images(self.path_load_images, nb_images, self.input_shape_vae[:2], random=random)
            print(f'{self.train_generator.shape[0]} images were loaded !')
            print(f'->Their shapes are {self.train_generator.shape[1:]}.')
    
    def create_encoder(self, logs=False):
        encoder_input = tf.keras.Input(shape=self.input_shape_vae, name='encoder_input')
        x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(encoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 3, activation='relu', strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 3, activation='relu', strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = tf.keras.Model(encoder_input, [z_mean, z_log_var, z], name='encoder')

        if logs:
            print(f'Encoder input shape: {encoder_input.shape}.')
            print(f'Latent Space dimension: {z_mean.shape}.')
            print('Encoder Summary:')
            print(encoder.summary())
            
        self.encoder = encoder
    
    def create_decoder(self, logs=False):
        decoder_input = tf.keras.Input(shape=(self.latent_dim,), name='decoder_input')
        x = layers.Dense(68 * 23 * 64, activation='relu')(decoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.Reshape((68, 23, 64))(x)
        x = layers.Conv2DTranspose(256, 3, activation='relu', strides=(1,1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(128, 3, activation='relu', strides=(2,2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(64, 3, activation='relu', strides=(2,2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(32, 3, activation='relu', strides=(2,2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        decoder_output = layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)

        # Crop layer to remove 3 pixels on height and width
        crop_layer = layers.Cropping2D(cropping=((0, 3), (0, 3)))
        decoder_output = crop_layer(decoder_output)

        decoder = tf.keras.Model(decoder_input, decoder_output, name='decoder')

        if logs:
            print(f'Decoder input shape: {decoder_input.shape}.')
            print(f'Decoder output shape: {decoder_output.shape}.')
            print('Decoder Summary:')
            print(decoder.summary())
            
        self.decoder = decoder
        
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction),
                    axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def save_coder_weights(self):
        # Save the weights to the given file path
        print('model_name:', self.model_name)
        self.encoder.save_weights(self.path_save_weights + f"/encoder.h5")
        self.decoder.save_weights(self.path_save_weights + f"/decoder.h5")
        
    def load_coder_weights(self, name):
        # Load the weights from the given file path
        self.encoder.load_weights(self.path_save_weights + f"/encoder.h5")
        self.decoder.load_weights(self.path_save_weights + f"/decoder.h5")
        
    def fit_images(self, optimizer, nb_epochs=50):
        
        # compile model
        self.compile(optimizer=optimizer)
        
        # define callbacks
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=1e-6)
        print(self.path_save_weights)
        checkpoint_path  = self.path_save_weights + f'{self.model_name}_model.h5'
        model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', save_best_only=True)        
        
        if self.debug_mode == False:
            history = self.fit(
                self.train_generator,
                epochs=nb_epochs,
                steps_per_epoch=self.train_generator.n // self.batch_size,
                callbacks=[early_stopping, model_checkpoint, reduce_lr]
            )        
        
        else:
            history = self.fit(
                self.train_generator,
                epochs=nb_epochs,
                batch_size=self.batch_size,
                callbacks=[early_stopping, model_checkpoint, reduce_lr]
            )
        self.save_coder_weights()
        self.history = history
    
    def proccess_and_save(self, nb_images, random, show):
        # create the reconstructed images
        images = get_images(self.path_load_images, nb_images, self.input_shape_vae[:2], random=random)
        z_mean, z_log_var, z = self.encoder.predict(images)
        reconstructed_image = self.decoder.predict(z)
        # save image and reconstructed images
        for i in range(nb_images):
            np.save(self.path_save_images + f'truth/{self.model_name}_{i}.npy', images[i])
            np.save(self.path_save_images + f'pred/{self.model_name}_{i}.npy', reconstructed_image[i])
        if show:
            if nb_images >=2:
                # plot images and reconstructed images
                nb_images_to_plot = min(nb_images, 10)
                fig, axes = plt.subplots(2, nb_images_to_plot, figsize=(15, 10))
                # images
                for i in range(nb_images_to_plot):
                    axes[0, i].imshow(images[i])
                    axes[0, i].axis('off') 
                # reconstructed images
                for i in range(nb_images_to_plot):
                    axes[1, i].imshow(reconstructed_image[i])
                    axes[1, i].axis('off')  # Turn off axis labels
                plt.show()
            else:
                print('nb_images should be at least 2!')
        
