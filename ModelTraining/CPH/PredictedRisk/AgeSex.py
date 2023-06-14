

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
target_df_file = dpath + 'PreprocessedData/TargetData/TargetData.csv'
cov_df = pd.read_csv(dpath + 'PreprocessedData/Covariates/PANEL.csv', usecols=['eid', 'Region_code', 'DM_AGE', 'DM_GENDER'])
cov_df.drop('Region_code', axis = 1, inplace = True)

my_formula = 'DM_AGE + C(DM_GENDER)'


target_code_df = pd.read_csv(dpath + 'Target_code.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()
fold_id_lst = [ele for ele in range(10)]


for target_code in target_code_lst:
    myoutdf = pd.DataFrame()
    for fold_id in fold_id_lst:
        try:
            prors_train = pd.read_csv(
                outpath + 'MLModeling/MLP/ResMLP/' + target_code + '/Train' + str(fold_id) + '_ProRS.csv')
            mydf_train = pd.merge(prors_train, cov_df, how='inner', on=['eid'])
            prors_test = pd.read_csv(
                outpath + 'MLModeling/MLP/ResMLP/' + target_code + '/Test' + str(fold_id) + '_ProRS.csv')
            mydf_test = pd.merge(prors_test, cov_df, how='inner', on=['eid'])
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(mydf_train, duration_col='Y_yrs', event_col='Y', formula=my_formula)
            pred_risk_test = 1 - cph.predict_survival_function(mydf_test, times=[15]).T
            pred_risk_test.columns = ['CPH_Risk']
            tmpout = pd.concat([prors_test, pred_risk_test], axis=1)
            myoutdf = pd.concat([myoutdf, tmpout], axis=0)
        except:
            pass
    print((target_code, roc_auc_score(myoutdf.Y, myoutdf.ProRS), roc_auc_score(myoutdf.Y, myoutdf.CPH_Risk)))
    myoutdf.to_csv(outpath + 'MLModeling/CPH/PredProbs/AgeSex/'+target_code+'.csv', index=False)


