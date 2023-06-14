

import numpy as np
import pandas as pd
import pickle
from Utility.Evaluation_Utilities import *
from sklearn.metrics import brier_score_loss, recall_score, roc_auc_score, average_precision_score
from Utility.Training_Utilities import *
from lifelines.utils import concordance_index
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
dpath1 = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/SuppData/sTables/sTable6/'

dpath2 = dpath1 + 'Data/10YEARS/'
target_code_df = pd.read_csv(dpath + 'Target_code.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()

myout_df = pd.DataFrame()

for target_code in target_code_lst:
    tmpdf = pd.read_csv(dpath2 + target_code + '.csv')
    myout_df[target_code] = tmpdf.iloc[-1, :]

myout_df = myout_df.T
myout_df['Disease_code'] = myout_df.index
myout_df = pd.merge(target_code_df, myout_df, how = 'left', on = ['Disease_code'])
myout_df.to_csv(dpath1 + '10YEARS.csv', index = False)

