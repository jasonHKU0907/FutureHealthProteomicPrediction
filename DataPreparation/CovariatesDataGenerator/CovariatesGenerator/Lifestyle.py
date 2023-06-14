

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

target_FieldID = ['20116-0.0', '1558-0.0', '1160-0.0']
mydf = read_data(target_FieldID, feature_df, eid_df)
mydf.rename(columns = {'20116-0.0': 'SMK_Status', '1558-0.0': 'ALC_Status', '1160-0.0': 'SleepDuration'}, inplace = True)

############## Smoking Status #############################
mydf['SMK_Status'].replace([np.nan, -3, 0, 1, 2], [np.nan, np.nan, 0, 0, 1], inplace = True)

############## Alcohol Status #############################
mydf['ALC_Status'].replace([np.nan, -3, 1, 2, 3, 4, 5, 6], [np.nan, np.nan, 1, 1, 1, 0, 0, 0], inplace = True)

############## Sleep duration #############################
mydf['HealthySleep'] = 0
mydf['HealthySleep'].loc[mydf.SleepDuration.isna() == True] = np.nan
mydf['HealthySleep'].loc[(mydf.SleepDuration >=7) & (mydf.SleepDuration <=9)] = 1

############## Physical activity & Healthy diet & Social Contact #############################
mydf_pa = pd.read_csv(outpath + 'Lifestyle_PA.csv', usecols = ['eid', 'RegularPA'])
mydf_diet = pd.read_csv(outpath + 'Lifestyle_Diet.csv', usecols = ['eid', 'HealthyDiet'])
mydf_social = pd.read_csv(outpath + 'Lifestyle_Social.csv', usecols = ['eid', 'SocialContact'])

mydf = pd.merge(mydf, mydf_pa, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, mydf_diet, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, mydf_social, how = 'left', on = ['eid'])


mydf.to_csv(outpath + 'LifeStyle.csv', index=False)

'''
for i in range(mydf.shape[1]):
    na_prop = np.round(mydf.describe().iloc[0,i]/502409*100,1)
    print((mydf.columns[i], na_prop))

'''
