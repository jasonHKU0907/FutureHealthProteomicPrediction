
# This file simply try to find the proportion of missingness in each individuals
# Individuals with Prop of NA>30% will be removed for further modeling and statistical analysis

import glob
import os
import numpy as np
import pandas as pd
import re
from sklearn.impute import KNNImputer
pd.options.mode.chained_assignment = None  # default='warn'
import time

def get_standardization(mydf):
    tmp_df = mydf.copy()
    for col in tmp_df.columns:
        ubd = tmp_df[col].quantile(0.995)
        lbd = tmp_df[col].quantile(0.005)
        tmp_df[col].iloc[tmp_df[col]>ubd] = ubd
        tmp_df[col].iloc[tmp_df[col]<lbd] = lbd
        tmp_df[col] = (tmp_df[col] - lbd) / (ubd - lbd)
    return tmp_df

def get_correlated_pros(pro_f, nb_f2select, corr_df):
    tmpdf = corr_df.copy()
    tmpdf_sorted = tmpdf.sort_values(by = [pro_f], ascending=False)
    f_selected = tmpdf_sorted.iloc[:(nb_f2select+1)].index.tolist()
    return f_selected

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'

pro_df = pd.read_csv(dpath + 'Proteomics_RAW/Proteomics_S1QC.csv')
pro_f_lst = pro_df.columns.tolist()[1:]
region_code_df = pd.read_csv(dpath + 'Eid_info_data.csv', usecols = ['eid', 'Region_code'])
pro_df = pd.merge(region_code_df, pro_df, how = 'right', on = 'eid')
region_code_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
demo_df = pd.read_csv(dpath + 'PreprocessedData/Covariates/PANEL.csv',
                      usecols = ['eid', 'DM_AGE', 'DM_GENDER', 'DM_ETH-W', 'DM_ETH-A', 'DM_ETH-B', 'DM_ETH-O', 'DM_TDI', 'DM_EDUC'])

demo_f = ['DM_AGE', 'DM_ETH-W', 'DM_ETH-A', 'DM_ETH-B', 'DM_ETH-O', 'DM_TDI', 'DM_EDUC']

mydf = pd.merge(pro_df, demo_df, how = 'inner', on = ['eid'])
del pro_df
del demo_df
del region_code_df

mydf[pro_f_lst] = get_standardization(mydf[pro_f_lst])


mydf_male = mydf.loc[mydf.DM_GENDER == 1]
mydf_male.reset_index(inplace = True)
mydf_male.drop(['DM_GENDER', 'index'], axis = 1, inplace = True)


mydf_female = mydf.loc[mydf.DM_GENDER == 0]
mydf_female.reset_index(inplace = True)
mydf_female.drop(['DM_GENDER', 'index'], axis = 1, inplace = True)


mydf_out_male = pd.DataFrame()
time0 = time.time()

for region_id in region_code_lst:
    tmpdf_train = mydf_male.loc[mydf_male.Region_code != region_id]
    tmpdf_test = mydf_male.loc[mydf_male.Region_code == region_id]
    tmpdf_test.reset_index(inplace = True)
    corr_df = tmpdf_train[pro_f_lst].corr().abs()
    i = 0
    for pro_f in pro_f_lst:
        f2select = get_correlated_pros(pro_f = pro_f, nb_f2select = 50, corr_df = corr_df)
        tmpdf_train_sub = tmpdf_train[f2select + demo_f]
        tmpdf_test_sub = tmpdf_test[f2select + demo_f]
        knn_imputer = KNNImputer(n_neighbors=50, weights="uniform")
        knn_imputer.fit(tmpdf_train_sub)
        tmpdf_test_sub_imputed = pd.DataFrame(knn_imputer.transform(tmpdf_test_sub))
        tmpdf_test_sub_imputed.columns = tmpdf_test_sub.columns.tolist()
        tmpdf_test[pro_f] = tmpdf_test_sub_imputed[pro_f]
        i+=1
        if i%500 == 0:
            print((region_id, i, time.time() - time0))
    mydf_out_male = pd.concat([mydf_out_male, tmpdf_test], axis=0)

mydf_out_male.to_csv(dpath + 'Proteomics_RAW/Proteomics_S2Male.csv', index = False)




mydf_out_female =pd.DataFrame()
time0 = time.time()

for region_id in region_code_lst:
    tmpdf_train = mydf_female.loc[mydf_female.Region_code != region_id]
    tmpdf_test = mydf_female.loc[mydf_female.Region_code == region_id]
    tmpdf_test.reset_index(inplace = True)
    corr_df = tmpdf_train[pro_f_lst].corr().abs()
    i = 0
    for pro_f in pro_f_lst:
        f2select = get_correlated_pros(pro_f = pro_f, nb_f2select = 50, corr_df = corr_df)
        tmpdf_train_sub = tmpdf_train[f2select + demo_f]
        tmpdf_test_sub = tmpdf_test[f2select + demo_f]
        knn_imputer = KNNImputer(n_neighbors=50, weights="uniform")
        knn_imputer.fit(tmpdf_train_sub)
        tmpdf_test_sub_imputed = pd.DataFrame(knn_imputer.transform(tmpdf_test_sub))
        tmpdf_test_sub_imputed.columns = tmpdf_test_sub.columns.tolist()
        tmpdf_test[pro_f] = tmpdf_test_sub_imputed[pro_f]
        i+=1
        if i%500 == 0:
            print((region_id, i, time.time() - time0))
    mydf_out_female = pd.concat([mydf_out_female, tmpdf_test], axis=0)

mydf_out_female.to_csv(dpath + 'Proteomics_RAW/Proteomics_S2Female.csv', index = False)


mydf_out_male = pd.read_csv(dpath + 'Proteomics_RAW/Proteomics_S2Male_9.csv')
mydf_out_female = pd.read_csv(dpath + 'Proteomics_RAW/Proteomics_S2Female_9.csv')
mydf_out = pd.concat([mydf_out_male, mydf_out_female], axis = 0)
mydf_out.sort_values(by = ['eid'], ascending = True, inplace = True)
mydf_out = mydf_out[['eid', 'Region_code'] + pro_f_lst]

mydf_out.to_csv(dpath + 'PreprocessedData/ProteomicsData/ProteomicsData_imp.csv', index = False)
