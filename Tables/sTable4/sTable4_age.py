

import numpy as np
import pandas as pd
import pickle
from Utility.Evaluation_Utilities import *
from Utility.Training_Utilities import *
from lifelines.utils import concordance_index
from scipy.stats.stats import pearsonr

pd.options.mode.chained_assignment = None  # default='warn'

def pearsonr_ci(x,y,alpha=0.05):
    r, p = pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi

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
    corr, pval, lo, hi = pearsonr_ci(mydf.ProRS, mydf.DM_AGE)
    out_lst.append([target_code, corr, lo, hi, pval])

myoutdf = pd.DataFrame(out_lst)
myoutdf.columns = ['Disease_code', 'Age_corr', 'Corr_lbd', 'Corr_ubd', 'p-value']

my_out_lst = []
for i in range(len(myoutdf)):
    my_mean = f'{myoutdf.Age_corr[i]:.2f}'
    my_lbd = f'{myoutdf.Corr_lbd[i]:.2f}'
    my_ubd = f'{myoutdf.Corr_ubd[i]:.2f}'
    my_out_lst.append(my_mean + ' [' + my_lbd + '-' + my_ubd + ']')

myoutdf['output'] = my_out_lst

myoutdf = pd.merge(target_code_df, myoutdf, how = 'left', on = ['Disease_code'])

myoutdf.to_csv(outpath1+'Age_correlation1.csv', index = False)

