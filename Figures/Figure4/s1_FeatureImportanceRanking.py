

import os
import numpy as np
import random
import pandas as pd
from Utility.Evaluation_Utilities import *
import shap
import time
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
target_df_file = dpath + 'PreprocessedData/TargetData/TargetData.csv'
target_code_df = pd.read_csv(dpath + 'Target_code.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()
fold_id_lst = [ele for ele in range(10)]

pro_df = pd.read_csv(dpath + 'PreprocessedData/ProteomicsData/ProteomicsData_imp.csv')
code_dict_df = pd.read_csv(dpath + 'ProCode.csv', usecols=['coding', 'Pro_code'])
code_dict_df['coding'] = code_dict_df['coding'].astype(str)
pro_code_df = pd.DataFrame({'coding':pro_df.columns[:-1]})
pro_code_df['coding'] = pro_code_df['coding'].astype(str)
rename_dict_df = pd.merge(pro_code_df, code_dict_df[['coding', 'Pro_code']], how = 'left', on = ['coding'])
rename_dict = dict(zip(rename_dict_df.coding, rename_dict_df.Pro_code))
pro_df.rename(columns = rename_dict, inplace = True)
pro_df.rename(columns={ pro_df.columns[0]: "eid" }, inplace = True)
pro_f_lst = pro_df.columns.tolist()[1:]
pro_df.to_csv(dpath + 'PreprocessedData/ProteomicsData/ProteomicsDataPlot.csv', index = False)

for target_code in target_code_lst:
    target_y, target_yrs = target_code + '_y', target_code + '_years'
    target_df = preprocess_target_individuals(target_df_file, target_code)
    mydf = pd.merge(pro_df, target_df, how = 'inner', on = ['eid'])
    X, y, y_yrs = mydf[pro_f_lst].copy(), mydf[target_y].copy(), mydf[target_yrs].copy()
    shap_imp_cv = np.zeros(len(pro_f_lst))
    for fold_id in fold_id_lst:
                mc_file = target_path + 'MLP_fold' + str(fold_id) + '.h5'
        train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
        test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
        X_train0, X_test = np.array(X.iloc[train_idx, :]), np.array(X.iloc[test_idx, :])
        y_train0, y_test = np.array(y.iloc[train_idx]), np.array(y.iloc[test_idx])
        best_model = load_model(mc_file)
        start_time = time.time()
        explainer = shap.DeepExplainer(best_model, X_train0)
        shap_values = explainer.shap_values(X_test)
        shap_values = np.abs(np.average(shap_values[0], axis=0))
        shap_imp_cv += shap_values / np.sum(shap_values)
    shap_imp_df = pd.DataFrame({'Pro_code': pro_f_lst, 'ShapValues_cv': shap_imp_cv / 10})
    shap_imp_df.sort_values(by='ShapValues_cv', ascending=False, inplace=True)
    shap_imp_df.to_csv(outpath + 'Plots/Figure3/Data/FeaImp_ResMLP/' + target_code + '.csv', index=False)

