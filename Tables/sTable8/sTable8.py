

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
    myout_df = mydf1[['Disease_code']]
    cidx_region_lst = ['C_idx_region0', 'C_idx_region1', 'C_idx_region2', 'C_idx_region3', 'C_idx_region4',
                       'C_idx_region5', 'C_idx_region6', 'C_idx_region7', 'C_idx_region8', 'C_idx_region9']
    tmpdf1 = mydf1[cidx_region_lst]
    tmpdf2 = mydf2[cidx_region_lst]
    tmpdf1[tmpdf1 <= 0.5] = np.nan
    tmpdf2[tmpdf2 <= 0.5] = np.nan
    p_val_lst = [ttest_ind(tmpdf1.iloc[i], tmpdf2.iloc[i], alternative = option, nan_policy = 'omit')[1] for i in range(len(tmpdf1))]
    myout_df['p_val'] = p_val_lst
    indicator_lst = []
    for i in range(len(myout_df)):
        pval = p_val_lst[i]
        if pval<0.001:
            indicator_lst.append('***')
        elif pval<0.01:
            indicator_lst.append('**')
        elif pval<0.05:
            indicator_lst.append('*')
        else:
            indicator_lst.append('')
    myout_df['sig_indicator'] = indicator_lst
    return myout_df

def get_cidx_out(mydf):
    myout_df = mydf[['Disease_code']]
    cidx_region_lst = ['C_idx_region0', 'C_idx_region1', 'C_idx_region2', 'C_idx_region3', 'C_idx_region4',
                       'C_idx_region5', 'C_idx_region6', 'C_idx_region7', 'C_idx_region8', 'C_idx_region9']
    tmpdf = mydf[cidx_region_lst]
    tmpdf[tmpdf <= 0.5] = np.nan
    tmpdf['cidx_mean'] = tmpdf.mean(axis=1)
    tmpdf['cidx_std'] = tmpdf.std(axis=1)
    tmpdf['cidx_lbd'] = tmpdf['cidx_mean'] - 1.96 * tmpdf['cidx_std'] / np.sqrt(10)
    tmpdf['cidx_ubd'] = tmpdf['cidx_mean'] + 1.96 * tmpdf['cidx_std'] / np.sqrt(10)
    cidx_out_lst = []
    for i in range(len(tmpdf)):
        my_mean = f'{tmpdf.cidx_mean.iloc[i]:.2f}'
        my_lbd = f'{tmpdf.cidx_lbd.iloc[i]:.2f}'
        my_ubd = f'{tmpdf.cidx_ubd.iloc[i]:.2f}'
        cidx_out = my_mean + ' [' + my_lbd + '-' + my_ubd +']'
        cidx_out_lst.append(cidx_out)
    myout_df['cidx_out'] = cidx_out_lst
    return myout_df

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/SuppData/sTables/sTable8/'
target_code_df = pd.read_csv('/Volumes/JasonWork/Projects/UKB_Proteomics/Data/Target_code.csv')

cph_df0 = pd.read_csv(dpath + 'MLModeling/CPH/ProRS/ProRS_Cidx.csv')
cph_df10 = pd.read_csv(dpath + 'MLModeling/CPH/AgeSex/AgeSex_Cidx.csv')
cph_df11 = pd.read_csv(dpath + 'MLModeling/CPH/Combo/ProRS_AgeSex_Cidx.csv')
cph_df20 = pd.read_csv(dpath + 'MLModeling/CPH/Serum/SERUM_Cidx.csv')
cph_df21 = pd.read_csv(dpath + 'MLModeling/CPH/Combo/ProRS_SERUM_Cidx.csv')
cph_df30 = pd.read_csv(dpath + 'MLModeling/CPH/PANEL/PANEL_Cidx.csv')
cph_df31 = pd.read_csv(dpath + 'MLModeling/CPH/Combo/ProRS_PANEL_Cidx.csv')

pvaldf10 = get_stats(cph_df0, cph_df10, option = 'two-sided')
pvaldf10.rename(columns = {'p_val': 'pval_ProRS_AgeSex', 'sig_indicator': 'sig_ProRS_AgeSex'}, inplace = True)

pvaldf11 = get_stats(cph_df11, cph_df0, option = 'two-sided')
pvaldf11.rename(columns = {'p_val': 'pval_ProRSAgeSex_ProRS', 'sig_indicator': 'sig_ProRSAgeSex_ProRS'}, inplace = True)

pvaldf111 = get_stats(cph_df11, cph_df10, option = 'two-sided')
pvaldf111.rename(columns = {'p_val': 'pval_ProRSAgeSex_AgeSex', 'sig_indicator': 'sig_ProRSAgeSex_AgeSex'}, inplace = True)


pvaldf20 = get_stats(cph_df0, cph_df20, option = 'two-sided')
pvaldf20.rename(columns = {'p_val': 'pval_ProRS_Serum', 'sig_indicator': 'sig_ProRS_Serum'}, inplace = True)

pvaldf21 = get_stats(cph_df21, cph_df0, option = 'two-sided')
pvaldf21.rename(columns = {'p_val': 'pval_ProRSSerum_ProRS', 'sig_indicator': 'sig_ProRSSerum_ProRS'}, inplace = True)

pvaldf222 = get_stats(cph_df21, cph_df20, option = 'two-sided')
pvaldf222.rename(columns = {'p_val': 'pval_ProRSSerum_Serum', 'sig_indicator': 'sig_ProRSSerum_Serum'}, inplace = True)



pvaldf30 = get_stats(cph_df0, cph_df30, option = 'two-sided')
pvaldf30.rename(columns = {'p_val': 'pval_ProRS_PANEL', 'sig_indicator': 'sig_ProRS_PANEL'}, inplace = True)

pvaldf31 = get_stats(cph_df31, cph_df0, option = 'two-sided')
pvaldf31.rename(columns = {'p_val': 'pval_ProRSPANEL_ProRS', 'sig_indicator': 'sig_ProRSPANEL_ProRS'}, inplace = True)

pvaldf333 = get_stats(cph_df31, cph_df30, option = 'two-sided')
pvaldf333.rename(columns = {'p_val': 'pval_ProRSPANEL_PANEL', 'sig_indicator': 'sig_ProRSPANEL_PANEL'}, inplace = True)


mydf0 = get_cidx_out(cph_df0)
mydf0.rename(columns = {'cidx_out': 'Cidx_ProRS'}, inplace = True)

mydf10 = get_cidx_out(cph_df10)
mydf10.rename(columns = {'cidx_out': 'Cidx_AgeSex'}, inplace = True)

mydf11 = get_cidx_out(cph_df11)
mydf11.rename(columns = {'cidx_out': 'Cidx_AgeSexProRS'}, inplace = True)

mydf20 = get_cidx_out(cph_df20)
mydf20.rename(columns = {'cidx_out': 'Cidx_Serum'}, inplace = True)

mydf21 = get_cidx_out(cph_df21)
mydf21.rename(columns = {'cidx_out': 'Cidx_SerumProRS'}, inplace = True)

mydf30 = get_cidx_out(cph_df30)
mydf30.rename(columns = {'cidx_out': 'Cidx_PANEL'}, inplace = True)

mydf31 = get_cidx_out(cph_df31)
mydf31.rename(columns = {'cidx_out': 'Cidx_PANELProRS'}, inplace = True)

myout_df = pd.merge(target_code_df, mydf0, how = 'left', on = ['Disease_code'])
myout_df = pd.merge(myout_df, mydf10, how = 'left', on = ['Disease_code'])
myout_df = pd.merge(myout_df, mydf11, how = 'left', on = ['Disease_code'])
myout_df = pd.merge(myout_df, mydf20, how = 'left', on = ['Disease_code'])
myout_df = pd.merge(myout_df, mydf21, how = 'left', on = ['Disease_code'])
myout_df = pd.merge(myout_df, mydf30, how = 'left', on = ['Disease_code'])
myout_df = pd.merge(myout_df, mydf31, how = 'left', on = ['Disease_code'])

myout_df = pd.merge(myout_df, pvaldf10, how = 'left', on = ['Disease_code'])
myout_df = pd.merge(myout_df, pvaldf11, how = 'left', on = ['Disease_code'])
myout_df = pd.merge(myout_df, pvaldf111, how = 'left', on = ['Disease_code'])
myout_df = pd.merge(myout_df, pvaldf20, how = 'left', on = ['Disease_code'])
myout_df = pd.merge(myout_df, pvaldf21, how = 'left', on = ['Disease_code'])
myout_df = pd.merge(myout_df, pvaldf222, how = 'left', on = ['Disease_code'])
myout_df = pd.merge(myout_df, pvaldf30, how = 'left', on = ['Disease_code'])
myout_df = pd.merge(myout_df, pvaldf31, how = 'left', on = ['Disease_code'])
myout_df = pd.merge(myout_df, pvaldf333, how = 'left', on = ['Disease_code'])

myout_df.to_csv(outpath + 'sTable8.csv', index = True)


