

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

target_FieldID = ['31-0.0', '884-0.0', '894-0.0', '904-0.0', '914-0.0']
mydf = read_data(target_FieldID, feature_df, eid_df)

'''
UK Biobank Touchscreen questionnaire at baseline; 
≥ 150 minutes moderate activity per week
≥ 75 minutes vigorous activity per week
≥ 150 minutes moderate + vigorous activity per week
moderate physical activity at least 5 days a week
vigorous activity once a week 
'''

####################################################################################################
##### Number of days/week of moderate physical activity 10+ minutes (Field ID 884)     #############
##### Duration of moderate activity (Field ID 894)                                     #############
##### Number of days/week of vigorous physical activity 10+ minutes (Field ID 904)     #############
##### Duration of vigorous activity (Field ID 914)                                     #############
####################################################################################################

mydf_male = mydf.loc[mydf['31-0.0'] == 1]
mydf_male['884-0.0'].replace([-1, -3, np.nan], np.nan, inplace = True)
mydf_male['894-0.0'].replace([-1, -3, np.nan], np.nan, inplace = True)
mydf_male['904-0.0'].replace([-1, -3, np.nan], np.nan, inplace = True)
mydf_male['914-0.0'].replace([-1, -3, np.nan], np.nan, inplace = True)

mydf_female = mydf.loc[mydf['31-0.0'] == 0]
mydf_female['884-0.0'].replace([-1, -3, np.nan], np.nan, inplace = True)
mydf_female['894-0.0'].replace([-1, -3, np.nan], np.nan, inplace = True)
mydf_female['904-0.0'].replace([-1, -3, np.nan], np.nan, inplace = True)
mydf_female['914-0.0'].replace([-1, -3, np.nan], np.nan, inplace = True)

mydf = pd.concat([mydf_male, mydf_female], axis = 0)
mydf.sort_values(by = ['eid'], inplace = True)

ph_level = np.zeros((len(mydf)))

for i in range(len(mydf)):
    if mydf['894-0.0'].iloc[i]*7 >= 150:
        ph_level[i] += 1
    if mydf['914-0.0'].iloc[i]*7 >= 75:
        ph_level[i] += 1
    if mydf['914-0.0'].iloc[i]*7 + mydf['914-0.0'].iloc[i]*7 >= 150:
        ph_level[i] += 1
    if mydf['884-0.0'].iloc[i] >= 5:
        ph_level[i] += 1
    if mydf['904-0.0'].iloc[i] >= 1:
        ph_level[i] += 1

mydf['RegularPA'] = pd.DataFrame(ph_level)
mydf['RegularPA'].iloc[mydf['RegularPA']>0] = 1
mydf['RegularPA'].value_counts()

mydf.to_csv(outpath + 'Lifestyle_PA.csv', index=False)

'''
for i in range(mydf.shape[1]):
    na_prop = np.round(mydf.describe().iloc[0,i]/502409*100,1)
    print((mydf.columns[i], na_prop))

'''
