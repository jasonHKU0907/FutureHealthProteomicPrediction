

import numpy as np
import pandas as pd
import pickle
from Utility.Evaluation_Utilities import *
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from statsmodels.stats.multitest import fdrcorrection
from mne.stats import bonferroni_correction
pd.options.mode.chained_assignment = None  # default='warn'

def preprocess_target_individuals(target_df_file, target_code):
    target_y0, target_yrs0, target_source0 = target_code[0] + '0_y', target_code[0] + '0_years', target_code[0] + '0_source'
    target_y, target_yrs, target_source = target_code + '_y', target_code + '_years', target_code + '_source'
    target_df = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y, target_yrs])
    if (('C' in target_code) | ('X' in target_code)):
        target_df0 = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y0, target_yrs0])
        rm_idx0 = target_df0.index[target_df0[target_yrs0] <= 0]
        rm_eid_lst = target_df0.iloc[rm_idx0].eid.tolist()
    else:
        target_df0 = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y0, target_yrs0, target_source0])
        rm_idx0 = target_df0.index[(target_df0[target_source0] >= 50) | (target_df0[target_yrs0] <= 0)]
        rm_eid_lst = target_df0.iloc[rm_idx0].eid.tolist()
    clean_df = target_df[~target_df.eid.isin(rm_eid_lst)]
    clean_df.reset_index(inplace=True)
    clean_df.drop('index', axis=1, inplace=True)
    return clean_df

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/'
pro_df = pd.read_csv(dpath + 'PreprocessedData/ProteomicsData/ProteomicsDataPlot.csv')
pro_f_lst = pro_df.columns.tolist()[1:]
cov_df = pd.read_csv(dpath + 'PreprocessedData/Covariates/PANEL.csv', usecols = ['eid', 'DM_AGE', 'DM_GENDER'])

target_df_file = dpath + 'PreprocessedData/TargetData/TargetData.csv'
target_code_df = pd.read_csv(dpath + 'Target_code.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()
fold_id_lst = [ele for ele in range(10)]

my_formula = 'target_pro + DM_AGE + C(DM_GENDER)'

for target_code in target_code_lst:
    target_y, target_yrs = target_code + '_y', target_code + '_years'
    target_df = preprocess_target_individuals(target_df_file, target_code)
    mydf = pd.merge(target_df, cov_df, how='inner', on=['eid'])
    mydf = pd.merge(mydf, pro_df, how = 'inner', on = ['eid'])
    myout_df = pd.DataFrame()
    for pro_f in pro_f_lst:
        tmpdf = mydf[[pro_f] + ['DM_AGE', 'DM_GENDER'] + [target_y, target_yrs]]
        tmpdf.rename(columns={pro_f: 'target_pro', target_code + '_y': 'target_y', target_code + '_years': 'target_years'}, inplace=True)
        cph = CoxPHFitter(penalizer = 0.05)
        cph.fit(tmpdf, duration_col='target_years', event_col='target_y', formula=my_formula)
        hr = cph.hazard_ratios_.target_pro
        lbd = np.exp(cph.confidence_intervals_).iloc[-1, 0]
        ubd = np.exp(cph.confidence_intervals_).iloc[-1, 1]
        pval = cph.summary.p.target_pro
        myout = pd.DataFrame([pro_f, hr, lbd, ubd, pval])
        myout_df = pd.concat((myout_df, myout.T), axis=0)
    myout_df.columns = ['Pro_code', 'HR', 'HR_Lower_CI', 'HR_Upper_CI', 'HR_p_val']
    _, p_f_fdr = fdrcorrection(myout_df.HR_p_val.fillna(1))
    _, p_f_bfi = bonferroni_correction(myout_df.HR_p_val.fillna(1), alpha=0.05)
    myout_df['HR_pval_fdr'] = p_f_fdr
    myout_df['HR_pval_bfi'] = p_f_bfi
    hr_out = ["%.2f" % round(myout_df.HR.iloc[i], 2) for i in range(len(myout_df))]
    myout_df['HR_rounded'] = hr_out
    out = ["%.2f" % round(myout_df.HR.iloc[i], 2) + ' [' + "%.2f" % round(myout_df.HR_Lower_CI.iloc[i], 2) + '-' + "%.2f" % round(myout_df.HR_Upper_CI.iloc[i], 2) + ']' for i in range(len(myout_df))]
    myout_df['HR_out'] = out
    myout_df.to_csv(outpath + 'Plots/Figure4/Data/CPH/' + target_code + '1.csv', index=False)


