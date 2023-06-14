

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

target_FieldID = ['93-0.0', '94-0.0', '4079-0.0', '4080-0.0', '50-0.0', '21002-0.0', '48-0.0', '21001-0.0', '23105-0.0']
mydf = read_data(target_FieldID, feature_df, eid_df)
sbp_na_idx = mydf.index[mydf['4080-0.0'].isnull()]
mydf.loc[sbp_na_idx, '4080-0.0'] = mydf.loc[sbp_na_idx, '93-0.0']
dbp_na_idx = mydf.index[mydf['4079-0.0'].isnull()]
mydf.loc[dbp_na_idx, '4079-0.0'] = mydf.loc[dbp_na_idx, '94-0.0']

mydf.sort_values(by = ['eid'], inplace = True)

mydf.rename(columns = {'4080-0.0': 'SBP', '4079-0.0': 'DBP', '50-0.0': 'Height', '21002-0.0': 'Weight',
                       '48-0.0': 'WaistCir', '21001-0.0': 'BMI', '23105-0.0': 'BMR'}, inplace = True)
mydf.to_csv(outpath + 'PhysicalMeasurements.csv', index=False)

'''
for i in range(mydf.shape[1]):
    na_prop = np.round(mydf.describe().iloc[0,i]/502409*100,1)
    print((mydf.columns[i], na_prop))

'''