

import numpy as np
import pandas as pd
import pickle
from Utility.Evaluation_Utilities import *
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/'
outpath = dpath + 'Results/SuppData/sTables/sTable9/'
target_code_df = pd.read_csv(dpath + 'Data/Target_code.csv')
target_code_lst = target_code_df.Disease_code.tolist()

myout_df = pd.DataFrame()

for target_code in target_code_lst:
    shap_tmpdf = pd.read_csv(dpath + 'Results/MLModeling/MLP/FeaImp_ResMLP/' + target_code + '.csv')
    tmp_dis = str(target_code_df.Disease[target_code_df.Disease_code == target_code].iloc[0])
    shap_tmpdf.sort_values(by = 'ShapValues_cv', ascending = False, inplace = True)
    top_pro = shap_tmpdf.Pro_code.tolist()[:15]
    top_shapval = shap_tmpdf.ShapValues_cv.tolist()[:15]
    top_shapval = [f'{ele:.3f}' for ele in top_shapval]
    top_out_lst = [str(top_pro[i]) + ' (' + top_shapval[i] + ')' for i in range(15)]
    tmpout_df = pd.DataFrame({tmp_dis: top_out_lst})
    myout_df = pd.concat([myout_df, tmpout_df], axis = 1)

myout_df = myout_df.T

myout_df.to_csv(outpath + 'sTable9.csv', index = True)


