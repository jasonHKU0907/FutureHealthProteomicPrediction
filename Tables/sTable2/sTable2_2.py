

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
pd.options.mode.chained_assignment = None  # default='warn'
from statsmodels.stats.multitest import fdrcorrection
from mne.stats import bonferroni_correction

dpath_raw = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/Covariates/RAW/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/SuppData/sTables/sTable2/'
mydf = pd.read_csv(dpath_raw + 'CovariatesPANEL_RAW.csv')
cov_dict_df = pd.read_csv(outpath + 'CovariateDict.csv')

cont_f_lst = ['DM_AGE', 'DM_TDI', 'DM_EDUC',
              'PM_BMI', 'PM_BMR', 'PM_WT', 'PM_HT', 'PM_WC', 'PM_DBP', 'PM_SBP', 'MH_NBMED',
              'BF_LP_TC', 'BF_LP_HDLC', 'BF_LP_LDLC', 'BF_LP_TRG', 'BF_GL_GL', 'BF_GL_HBA1C',
              'BF_IF_CRP', 'BF_EC_IGF1', 'BF_EC_SHBG', 'BF_LV_AP', 'BF_LV_ALT',
              'BF_LV_AST', 'BF_LV_GGT', 'BF_RE_ALB', 'BF_RE_TPRO', 'BF_RE_CR',
              'BF_RE_CYSC', 'BF_RE_UREA', 'BF_BC_WBC', 'BF_BC_RBC', 'BF_BC_HB',
              'BF_BC_HCT', 'BF_BC_MCV', 'BF_BC_MCH', 'BF_BC_PLT']

target_y = 'DM_GENDER'

result_cont = []

for ele_f in cont_f_lst:
    tmpdf = mydf[[ele_f, target_y]].copy()
    tmpdf_pos = tmpdf.loc[mydf[target_y] == 1]
    tmpdf_neg = tmpdf.loc[mydf[target_y] == 0]
    median_all = np.round(tmpdf[ele_f].median(), 1)
    lqr_all = np.round(tmpdf[ele_f].quantile(0.25), 1)
    uqr_all = np.round(tmpdf[ele_f].quantile(0.75), 1)
    median_pos = np.round(tmpdf_pos[ele_f].median(), 1)
    lqr_pos = np.round(tmpdf_pos[ele_f].quantile(0.25), 1)
    uqr_pos = np.round(tmpdf_pos[ele_f].quantile(0.75), 1)
    median_neg = np.round(tmpdf_neg[ele_f].median(), 1)
    lqr_neg = np.round(tmpdf_neg[ele_f].quantile(0.25), 1)
    uqr_neg = np.round(tmpdf_neg[ele_f].quantile(0.75), 1)
    myout_tmp = [ele_f,
                 str(median_all) + ' [' + str(lqr_all) + '-' + str(uqr_all) + ']',
                 str(median_neg) + ' [' + str(lqr_neg) + '-' + str(uqr_neg) + ']',
                 str(median_pos) + ' [' + str(lqr_pos) + '-' + str(uqr_pos) + ']']
    result_cont.append(myout_tmp)

result_cont_df = pd.DataFrame(result_cont)
result_cont_df.columns = ['Covariate_code', 'All', 'Female', 'Male']

myout_df = pd.merge(result_cont_df, cov_dict_df, how = 'right', on = ['Covariate_code'])
myout_df.to_csv(outpath + 'sTable2_Continous.csv', index = True)
