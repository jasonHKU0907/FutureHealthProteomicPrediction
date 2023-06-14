

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
pd.options.mode.chained_assignment = None  # default='warn'
from statsmodels.stats.multitest import fdrcorrection
from mne.stats import bonferroni_correction

def get_cate_output(myout_tbl):
    myout_tbl_prop = myout_tbl.copy()
    myout_tbl_prop.iloc[:, 0] = np.round(myout_tbl_prop.iloc[:, 0] / myout_tbl.iloc[:, 0].sum() * 100, 1)
    myout_tbl_prop.iloc[:, 1] = np.round(myout_tbl_prop.iloc[:, 1] / myout_tbl.iloc[:, 1].sum() * 100, 1)
    myout_tbl_prop.iloc[:, 2] = np.round(myout_tbl_prop.iloc[:, 2] / myout_tbl.iloc[:, 2].sum() * 100, 1)
    myout_tbl_full = myout_tbl.copy()
    for i in range(len(myout_tbl_full)):
        for j in range(3):
            myout_tbl_full.iloc[i, j] = str(myout_tbl_full.iloc[i, j]) + ' (' + str(myout_tbl_prop.iloc[i, j]) + '%)'
    return myout_tbl_full


dpath_raw = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/Covariates/RAW/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/SuppData/sTables/sTable2/'
mydf = pd.read_csv(dpath_raw + 'CovariatesPANEL_RAW.csv')
cov_dict_df = pd.read_csv(outpath + 'CovariateDict.csv')

cate_f_lst = ['DM_ETH', 'LS_ALC', 'LS_SMK', 'LS_PA', 'LS_DIET', 'LS_SOC', 'LS_SLP',
              'DH_ILL', 'DH_DIAB', 'DH_HYPT', 'MH_CHOL', 'MH_BP', 'MH_INSU',
              'FH_DIAB', 'FH_HBP', 'FH_DM', 'FH_HD']

target_y = 'DM_GENDER'

result_cate = pd.DataFrame()

for ele_f in cate_f_lst:
    tmpdf = mydf[[ele_f, target_y]].copy()
    mytbl_all = pd.DataFrame(tmpdf[ele_f].value_counts())
    mytbl = pd.crosstab(tmpdf[ele_f], mydf[target_y], dropna=True)
    myout_tbl = pd.concat((mytbl_all, mytbl), axis = 1)
    myout_tbl.columns = ['All', 'Female', 'Male']
    myout_tbl.index = [ele_f + '_' + str(ele) for ele in myout_tbl.index]
    myout_tbl = get_cate_output(myout_tbl)
    result_cate = pd.concat((result_cate, myout_tbl), axis = 0)



Covariate_code_lst = [ele.split('_')[0] + '_' + ele.split('_')[1] for ele in result_cate.index]
result_cate['Covariate_code_raw'] = result_cate.index.tolist()
result_cate['Covariate_code'] = Covariate_code_lst
myout_df = pd.merge(result_cate, cov_dict_df, how = 'right', on = ['Covariate_code'])
myout_df.to_csv(outpath + 'sTable2_Categorical.csv', index = True)

