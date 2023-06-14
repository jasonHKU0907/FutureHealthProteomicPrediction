


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

col_lst = mydf.columns.to_list()

na_info_lst = []
for col in col_lst:
    nb_item = mydf[col].describe().iloc[0]
    nb_na = int(len(mydf) - nb_item)
    prop_na = str(np.round(nb_na/len(mydf)*100,2)) + '%'
    na_info_lst.append([col, nb_na, prop_na])

na_info_df = pd.DataFrame(na_info_lst)
na_info_df.columns = ['Covariate_code', 'Nb_NA', 'Prop_NA']

myout_df = pd.merge(na_info_df, cov_dict_df, how = 'right', on = ['Covariate_code'])
myout_df.to_csv(outpath + 'sTable2_NA.csv', index = True)


