

import os
import numpy as np
import random
import pandas as pd
from Utility.Evaluation_Utilities import *
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix
import shap
import time
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/'
target_code_df = pd.read_csv(dpath + 'Target_code.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()

target_code_lst = ['A0', 'C0', 'D0', 'E0', 'F0', 'G0', 'H0', 'H1', 'I0', 'J0', 'K0', 'L0', 'M0', 'N0', 'X0']

my_pro_lst = []
for target_code in target_code_lst:
    my_imp_df = pd.read_csv(outpath + 'Plots/Figure4/Data/FeaImp_ResMLP/' + target_code + '.csv')
    my_imp_df.sort_values('ShapValues_cv', ascending = False, inplace = True)
    my_pro_lst += my_imp_df.Pro_code[:15].tolist()

print(len(my_pro_lst))
print(len(list(set(my_pro_lst))))
my_pro_lst = list(set(my_pro_lst))

myoutdf = my_imp_df[['Pro_code']]

for target_code in target_code_lst:
    my_imp_df = pd.read_csv(outpath + 'Plots/Figure4/Data/FeaImp_ResMLP/' + target_code + '.csv')
    my_imp_df.sort_values('ShapValues_cv', ascending=False, inplace=True)
    tmp_pro_lst = my_imp_df.Pro_code[:15].tolist()
    top_pro_df = my_imp_df.loc[my_imp_df.Pro_code.isin(my_pro_lst)]
    top_pro_df.rename(columns = {'ShapValues_cv': target_code + '_shap'}, inplace = True)
    top15_indicator = [1 if ele in tmp_pro_lst else 0 for ele in my_pro_lst]
    select_df = pd.DataFrame({'Pro_code': my_pro_lst, target_code + '_top15': top15_indicator})
    myoutdf = pd.merge(myoutdf, top_pro_df, how = 'inner', on = ['Pro_code'])
    myoutdf = pd.merge(myoutdf, select_df, how = 'inner', on = ['Pro_code'])


top15_lst = [target_code+'_top15' for target_code in target_code_lst]
shap_lst = [target_code+'_shap' for target_code in target_code_lst]

myoutdf['ImpSum'] = myoutdf[shap_lst].sum(axis = 1)
myoutdf['Top15_nb'] = myoutdf[top15_lst].sum(axis = 1)
myoutdf.sort_values(by='ImpSum', ascending=False, inplace=True)
myoutdf.sort_values(by='Top15_nb', ascending=False, inplace=True)
myoutdf.to_csv(outpath + 'Plots/Figure4/Data/FeaImpRanking.csv')





