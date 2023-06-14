

import numpy as np
import pandas as pd
import pickle
from Utility.Evaluation_Utilities import *
from lifelines import CoxPHFitter
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix
from scipy.stats import ttest_ind
pd.options.mode.chained_assignment = None  # default='warn'
from lifelines.utils import concordance_index


def get_stats(mydf1, mydf2, option = 'two-sided'):
    myout_df = mydf1[['Disease', 'ICD_10', 'Disease_code']]
    tmpdf1 = mydf1[['C_idx_region0', 'C_idx_region1', 'C_idx_region2', 'C_idx_region3', 'C_idx_region4',
                     'C_idx_region5', 'C_idx_region6', 'C_idx_region7', 'C_idx_region8', 'C_idx_region9']]
    tmpdf2 = mydf2[['C_idx_region0', 'C_idx_region1', 'C_idx_region2', 'C_idx_region3', 'C_idx_region4',
                     'C_idx_region5', 'C_idx_region6', 'C_idx_region7', 'C_idx_region8', 'C_idx_region9']]
    tmpdf1[tmpdf1 < 0.5] = np.nan
    tmpdf2[tmpdf2 < 0.5] = np.nan
    diff_df_region = tmpdf2 - tmpdf1
    myout_df['diff_mean'] = diff_df_region.mean(axis = 1)
    myout_df['diff_std'] = diff_df_region.std(axis = 1)
    myout_df['diff_lbd'] = myout_df['diff_mean'] - 1.96*myout_df['diff_std']/np.sqrt(10)
    myout_df['diff_ubd'] = myout_df['diff_mean'] + 1.96*myout_df['diff_std']/np.sqrt(10)
    p_val_lst = [ttest_ind(tmpdf1.iloc[i], tmpdf2.iloc[i], alternative = option, nan_policy = 'omit')[1] for i in range(len(tmpdf1))]
    myout_df['p_val'] = p_val_lst
    return myout_df


dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/'
cph_df0 = pd.read_csv(dpath + 'MLModeling/CPH/ProRS/ProRS_Cidx.csv')
cph_df10 = pd.read_csv(dpath + 'MLModeling/CPH/AgeSex/AgeSex_Cidx.csv')
cph_df11 = pd.read_csv(dpath + 'MLModeling/CPH/Combo/ProRS_AgeSex_Cidx.csv')
cph_df20 = pd.read_csv(dpath + 'MLModeling/CPH/Serum/SERUM_Cidx.csv')
cph_df21 = pd.read_csv(dpath + 'MLModeling/CPH/Combo/ProRS_SERUM_Cidx.csv')
cph_df30 = pd.read_csv(dpath + 'MLModeling/CPH/PANEL/PANEL_Cidx.csv')
cph_df31 = pd.read_csv(dpath + 'MLModeling/CPH/Combo/ProRS_PANEL_Cidx.csv')

mydf10 = get_stats(cph_df0, cph_df10, option = 'two-sided')
mydf11 = get_stats(cph_df0, cph_df11, option = 'two-sided')

mydf20 = get_stats(cph_df0, cph_df20, option = 'two-sided')
mydf21 = get_stats(cph_df0, cph_df21, option = 'two-sided')

mydf30 = get_stats(cph_df0, cph_df30, option = 'two-sided')
mydf31 = get_stats(cph_df0, cph_df31, option = 'two-sided')

mydf10.to_csv(dpath + 'Plots/Figure3/Data/Base.csv', index = False)
mydf11.to_csv(dpath + 'Plots/Figure3/Data/BaseCombo.csv', index = False)
mydf20.to_csv(dpath + 'Plots/Figure3/Data/Serum.csv', index = False)
mydf21.to_csv(dpath + 'Plots/Figure3/Data/SerumCombo.csv', index = False)
mydf30.to_csv(dpath + 'Plots/Figure3/Data/PANEL.csv', index = False)
mydf31.to_csv(dpath + 'Plots/Figure3/Data/PANELCombo.csv', index = False)


########################################################################
####################### Manipulation ###################################
########################################################################


a = pd.read_csv(dpath + 'Plots/Figure3/Data/BaseProRS.csv', usecols = ['Disease'])
b = pd.read_csv(dpath + 'Plots/Figure3/Data/SerumProRS.csv')
b = pd.merge(a, b, how = 'left', on = ['Disease'])
b.to_csv(dpath + 'Plots/Figure3/Data/SerumProRS.csv', index = False)

c = pd.read_csv(dpath + 'Plots/Figure3/Data/PANELProRS.csv')
c = pd.merge(a, c, how = 'left', on = ['Disease'])
c.to_csv(dpath + 'Plots/Figure3/Data/PANELProRS.csv', index = False)




