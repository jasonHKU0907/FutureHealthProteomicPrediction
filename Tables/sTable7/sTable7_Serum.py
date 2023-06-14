

import numpy as np
import pandas as pd
import pickle
from Utility.Evaluation_Utilities import *
from lifelines import CoxPHFitter
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix
pd.options.mode.chained_assignment = None  # default='warn'
from lifelines.utils import concordance_index

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/'
outpath1 = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/SuppData/sTables/sTable7/'
target_code_df = pd.read_csv(dpath + 'Target_code.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()
cov_f_lst = ['BF_LP_TC', 'BF_LP_HDLC', 'BF_LP_LDLC', 'BF_LP_TRG', 'BF_GL_GL',
             'BF_GL_HBA1C', 'BF_IF_CRP', 'BF_EC_IGF1', 'BF_EC_SHBG', 'BF_LV_AP',
             'BF_LV_ALT', 'BF_LV_AST', 'BF_LV_GGT', 'BF_RE_ALB', 'BF_RE_TPRO',
             'BF_RE_CR', 'BF_RE_CYSC', 'BF_RE_UREA', 'BF_BC_WBC', 'BF_BC_RBC',
             'BF_BC_HB', 'BF_BC_HCT', 'BF_BC_MCV', 'BF_BC_MCH', 'BF_BC_PLT']
cov_df = pd.read_csv(dpath + 'PreprocessedData/Covariates/PANEL.csv', usecols = ['eid'] + cov_f_lst)
fold_id_lst = [ele for ele in range(10)]


myformula = 'BF_LP_TC + BF_LP_HDLC + BF_LP_LDLC + BF_LP_TRG + '\
             'BF_GL_GL + BF_GL_HBA1C + BF_IF_CRP + BF_EC_IGF1 + BF_EC_SHBG + '\
             'BF_LV_AP + BF_LV_ALT + BF_LV_AST + BF_LV_GGT + '\
             'BF_RE_ALB + BF_RE_TPRO + BF_RE_CR + BF_RE_CYSC + BF_RE_UREA + '\
             'BF_BC_WBC + BF_BC_RBC + BF_BC_HB + BF_BC_HCT + BF_BC_MCV + BF_BC_MCH + BF_BC_PLT + '\
             'ProRS'

pval_lst, hr_lst, lbd_lst, ubd_lst = [], [], [], []

for target_code in target_code_lst:
    mydf = pd.DataFrame()
    for fold_id in fold_id_lst:
        prors_test = pd.read_csv(outpath + 'MLModeling/MLP/ResMLP/' + target_code + '/Test' + str(fold_id) + '_ProRS.csv')
        mydf = pd.concat([mydf, prors_test], axis=0)
    mydf = pd.merge(mydf, cov_df, how = 'left', on = ['eid'])
    mydf.reset_index(inplace = True, drop = True)
    mydf['ProRS'] = (mydf['ProRS'] - mydf['ProRS'].mean())/mydf['ProRS'].std()
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(mydf, duration_col='Y_yrs', event_col='Y', formula=myformula)
    pval_lst.append(cph.summary.p.ProRS)
    hr_lst.append(cph.hazard_ratios_.ProRS)
    lbd_lst.append(np.exp(cph.confidence_intervals_.iloc[25, 0]))
    ubd_lst.append(np.exp(cph.confidence_intervals_.iloc[25, 1]))

hr_out_lst, p_out_lst1, p_out_lst2 = [], [], []
for i in range(len(hr_lst)):
    hr = f'{hr_lst[i]:.2f}'
    lbd = f'{lbd_lst[i]:.2f}'
    ubd = f'{ubd_lst[i]:.2f}'
    hr_out_lst.append(hr + ' [' + lbd + '-' + ubd + ']')
    if pval_lst[i]<0.001:
        p_out_lst1.append('<0.001')
    else:
        p_out_lst1.append(np.round(pval_lst[i], 3))

p_out_lst2 = ["{:.2e}".format(pval) for pval in pval_lst]
p_out_lst2 = [pval.split('e')[0] + 'x10' + pval.split('e')[1] for pval in p_out_lst2]


myout_df = pd.DataFrame([target_code_lst, hr_lst, lbd_lst, ubd_lst, pval_lst, hr_out_lst, p_out_lst1, p_out_lst2]).T
myout_df.columns = ['Disease_code', 'HR', 'HR_lbd', 'HR_ubd', 'Pval', 'HR_out', 'Pval_out1', 'Pval_out2']
myout_df = pd.merge(target_code_df, myout_df, how = 'left', on = ['Disease_code'])

myout_df.to_csv(outpath1 + 'ProRS_HR_SERUM.csv')


