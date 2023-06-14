

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

target_FieldID = ['709-0.0', '1031-0.0', '6160-0.0']
mydf = read_data(target_FieldID, feature_df, eid_df)

######################################################################################################
####################################   Social  #######################################################
######################################################################################################

'''
The social contact index was assessed using information on the number in the household, 
frequency of friend/family visits, 
and participation in leisure/social activity. 

1) number in the household: how many people are living together in your household 
   (1 point was given for living alone)
2) frequency of friend/family visits: how often do you visit friends or family or have them visit you 
  (1 point was given for answering about once a month, once every few months, never or almost never, 
   or no friends or family outside household)
3) which of the following (sports club or gym, pub or social club, religious group, adult education class, other group activity)
   do you engage in once a week or more often 
   (1 point was given for answering none of the above).
'''

mydf['709-0.0'].replace([-1, -3, np.nan], np.nan, inplace = True)
mydf['1031-0.0'].replace([-1, -3, np.nan], np.nan, inplace = True)

mydf['LiveAlone'] = 0
mydf['LiveAlone'].loc[mydf['709-0.0'].isna() == True] = np.nan
mydf['LiveAlone'].loc[mydf['709-0.0'] <=1] = 1

mydf['InFreq_vits'] = 0
mydf['InFreq_vits'].loc[mydf['1031-0.0'].isna() == True] = np.nan
mydf['InFreq_vits'].loc[mydf['1031-0.0'] >=4] = 1

mydf['6160-0.0'].fillna('NA', inplace = True)

for i in range(len(mydf)):
    item = mydf['6160-0.0'].iloc[i]
    if '-3' in item:
        mydf['6160-0.0'].iloc[i] = 'NA'

no_activity = np.zeros(len(mydf)).tolist()

for i in range(len(mydf)):
    item_lst = mydf['6160-0.0'].iloc[i]
    if item_lst == 'NA':
        no_activity[i] = -1
    if '-7' in item_lst:
        no_activity[i] = 1

mydf['No_Activity'] = pd.DataFrame(no_activity)
mydf['No_Activity'].replace(-1, np.nan, inplace = True)

mydf['SocialIsolation'] = mydf['LiveAlone'] + mydf['InFreq_vits'] + mydf['No_Activity']
mydf['SocialContact'] = 3- mydf['SocialIsolation']

mydf.to_csv(outpath + '/Lifestyle_Social.csv', index=False)
