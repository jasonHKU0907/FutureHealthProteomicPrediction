

import numpy as np
import pandas as pd
import pickle
from Utility.Evaluation_Utilities import *
from Utility.Training_Utilities import *
import statsmodels.api as sm

pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/'
outpath1 = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/SuppData/sTables/sTable4/'
target_code_df = pd.read_csv(dpath + 'Target_code.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()
my_cov_df = pd.read_csv(dpath + 'PreprocessedData/Covariates/PANEL.csv', usecols=['eid', 'DM_AGE', 'DM_GENDER'])
fold_id_lst = [ele for ele in range(10)]

out_lst = []
for target_code in target_code_lst:
    mydf = pd.DataFrame()
    for fold_id in fold_id_lst:
        tmpdf = pd.read_csv(outpath + 'MLModeling/MLP/ResMLP/' + target_code + '/Test' + str(fold_id) + '_ProRS.csv')
        mydf = pd.concat([mydf, tmpdf], axis = 0)
    mydf = pd.merge(mydf, my_cov_df, how = 'left', on = ['eid'])
    mydf['ProRS'] = mydf.ProRS*100
    mydf_male = mydf.loc[mydf.DM_GENDER == 1]
    mydf_female = mydf.loc[mydf.DM_GENDER == 0]
    median_all = mydf.ProRS.median()
    lqr_all = mydf.ProRS.quantile(0.25)
    uqr_all = mydf.ProRS.quantile(0.75)
    median_male = mydf_male.ProRS.median()
    lqr_male = mydf_male.ProRS.quantile(0.25)
    uqr_male = mydf_male.ProRS.quantile(0.75)
    median_female = mydf_female.ProRS.median()
    lqr_female = mydf_female.ProRS.quantile(0.25)
    uqr_female = mydf_female.ProRS.quantile(0.75)
    _, pval = ttest_ind(mydf_male.ProRS, mydf_female.ProRS, nan_policy='omit')
    myout_tmp = [target_code, pval,
                 median_all, lqr_all, uqr_all,
                 median_male, lqr_male, uqr_male,
                 median_female, lqr_female, uqr_female]
    out_lst.append(myout_tmp)

myoutdf = pd.DataFrame(out_lst)
myoutdf.columns = ['Disease_code', 'p-value', 'median_all', 'lqr_all', 'uqr_all',
                   'median_male', 'lqr_male', 'uqr_male',
                   'median_female', 'lqr_female', 'uqr_female']

myout_all_lst = []
for i in range(len(myoutdf)):
    my_mean = f'{myoutdf.median_all[i]:.1f}'
    my_lbd = f'{myoutdf.lqr_all[i]:.1f}'
    my_ubd = f'{myoutdf.uqr_all[i]:.1f}'
    myout_all_lst.append(my_mean + ' [' + my_lbd + '-' + my_ubd + ']')

myout_male_lst = []
for i in range(len(myoutdf)):
    my_mean = f'{myoutdf.median_male[i]:.1f}'
    my_lbd = f'{myoutdf.lqr_male[i]:.1f}'
    my_ubd = f'{myoutdf.uqr_male[i]:.1f}'
    myout_male_lst.append(my_mean + ' [' + my_lbd + '-' + my_ubd + ']')

myout_female_lst = []
for i in range(len(myoutdf)):
    my_mean = f'{myoutdf.median_female[i]:.1f}'
    my_lbd = f'{myoutdf.lqr_female[i]:.1f}'
    my_ubd = f'{myoutdf.uqr_female[i]:.1f}'
    myout_female_lst.append(my_mean + ' [' + my_lbd + '-' + my_ubd + ']')

myoutdf['output_all'] = myout_all_lst
myoutdf['output_male'] = myout_male_lst
myoutdf['output_female'] = myout_female_lst

myoutdf = pd.merge(target_code_df, myoutdf, how = 'left', on = ['Disease_code'])

myoutdf.to_csv(outpath1+'Gender_ttest1.csv', index = False)

