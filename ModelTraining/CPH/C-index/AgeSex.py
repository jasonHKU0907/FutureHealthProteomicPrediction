

import numpy as np
import pandas as pd
import pickle
from Utility.Evaluation_Utilities import *
from lifelines import CoxPHFitter
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix
pd.options.mode.chained_assignment = None  # default='warn'
from lifelines.utils import concordance_index

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
target_df_file = dpath + 'PreprocessedData/TargetData/TargetData.csv'
cov_df = pd.read_csv(dpath + 'PreprocessedData/Covariates/PANEL.csv', usecols= ['eid', 'DM_AGE', 'DM_GENDER'])
cov_f_lst = cov_df.columns.tolist()[1:]
target_code_df = pd.read_csv(dpath + 'Target_code.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()
fold_id_lst = [ele for ele in range(10)]

out_cidx_lst = []

for target_code in target_code_lst:
    target_y, target_yrs = target_code + '_y', target_code + '_years'
    target_df = preprocess_target_individuals(target_df_file, target_code)
    mydf = pd.merge(cov_df, target_df, how = 'inner', on = ['eid'])
    mydf.rename(columns ={target_code + '_y': 'target_y', target_code + '_years': 'target_years'}, inplace = True)
    cidx_lst = []
    for fold_id in fold_id_lst:
        train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
        test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
        mydf_train = mydf.iloc[train_idx].copy()
        mydf_test = mydf.iloc[test_idx].copy()
        cph = CoxPHFitter()
        my_formula = "DM_AGE + DM_GENDER"
        cph.fit(mydf_train, duration_col='target_years', event_col='target_y', formula=my_formula)
        cidx_lst.append(concordance_index(mydf_test['target_years'], -cph.predict_partial_hazard(mydf_test), mydf_test['target_y']))
    out_cidx_lst.append([target_code, np.mean(cidx_lst), np.std(cidx_lst)] + cidx_lst)
    out_df = pd.DataFrame(out_cidx_lst)
    out_df.columns = ['Disease_code', 'C_idx_mean', 'C_idx_std'] + ['C_idx_region' + str(ele) for ele in fold_id_lst]
    out_df = pd.merge(target_code_df, out_df, how='left', on=['Disease_code'])
    out_df.to_csv(outpath + 'MLModeling/CPH/AgeSex/AgeSex_Cidx.csv', index=False)
    print((target_code, np.mean(cidx_lst)))



