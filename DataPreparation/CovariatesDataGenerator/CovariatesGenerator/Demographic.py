

import glob
import os
import numpy as np
import pandas as pd
import re

def read_data(FieldID_lst, feature_df, eid_df):
    subset_df = feature_df[feature_df['Field_ID'].isin(FieldID_lst)]
    subset_dict = {k: ['eid'] + g['Field_ID'].tolist() for k, g in subset_df.groupby('Subset_ID')}
    subset_lst = list(subset_dict.keys())
    my_df = eid_df
    for subset_id in subset_lst:
        tmp_dir = dpath + 'UKB_subset_' + str(subset_id) + '.csv'
        tmp_f = subset_dict[subset_id]
        tmp_df = pd.read_csv(tmp_dir, usecols=tmp_f)
        my_df = pd.merge(my_df, tmp_df, how='inner', on=['eid'])
    return my_df

dpath = '/Volumes/JasonWork/Dataset/UKB_Tabular_merged_10/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/Covariates/RAW/'
feature_df = pd.read_csv(dpath + 'UKB_FieldID_Subset.csv')
eid_df = pd.read_csv(dpath + 'UKB_eid.csv')

target_FieldID = ['21022-0.0', '31-0.0', '21000-0.0', '189-0.0']
mydf = read_data(target_FieldID, feature_df, eid_df)

mydf['21022-0.0'].replace(np.nan, int(mydf['21022-0.0'].median()), inplace = True)
mydf['21000-0.0'].replace([1002, 1003, 3002, 3003, 5,    3004, 4002, 4003, 4,    1,    2001, 2002, 2003, 2004, 3,  2, -1,      -3],
                          [1001, 1001, 3001, 3001, 3001, 3001, 4001, 4001, 4001, 1001, 6,    6,    6,    6,    6,  6,  np.nan, np.nan], inplace = True)
mydf['21000-0.0'].replace([1001, 3001, 4001, 6], [1, 2, 3, 4], inplace = True)
#mydf['21000-0.0'].value_counts()
mydf.rename(columns = {'21022-0.0': 'Age', '31-0.0': 'Gender', '21000-0.0': 'Ethnicity', '189-0.0': 'TDI'}, inplace = True)
edu_df = pd.read_csv(outpath + 'Education.csv', usecols = ['eid', 'Education'])
mydf = pd.merge(mydf, edu_df, how = 'left', on = ['eid'])
mydf.sort_values(by = ['eid'], inplace = True)

mydf.to_csv(outpath + 'DemographicInfo.csv', index=False)

'''
for i in range(mydf.shape[1]):
    na_prop = np.round(mydf.describe().iloc[0,i]/502409*100,1)
    print((mydf.columns[i], na_prop))

'''
