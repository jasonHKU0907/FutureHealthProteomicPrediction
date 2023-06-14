

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

target_FieldID = ['31-0.0', '6177-0.0', '6153-0.0']
mydf = read_data(target_FieldID, feature_df, eid_df)

'''
derived from 6177-0.0 (male only) and 6153-0.0 (female only)
med_chol: take cholesterol lowering medication, 1: yes; 0: no; -1: missing
med_bp: take blood pressure medication, 1: yes; 0: no; -1: missing
med_insu: take insulin, 1: yes; 0: no; -1: missing
'''

######################################################################################################
####################################      Male     ###################################################
######################################################################################################

med_male = mydf.loc[mydf['31-0.0'] == 1][['eid', '6177-0.0']]
med_male['6177-0.0'].fillna('NA', inplace = True)
med_male['6177-0.0'].replace(['[-1]', '[-3]'], ['NA', 'NA'], inplace = True)
med_male_chol = np.zeros((med_male.shape[0])).tolist()
med_male_bp = np.zeros((med_male.shape[0])).tolist()
med_male_insu = np.zeros((med_male.shape[0])).tolist()

for i in range(med_male.shape[0]):
    item_lst = med_male['6177-0.0'].iloc[i]
    if item_lst == 'NA':
        med_male_chol[i], med_male_bp[i], med_male_insu[i] = -1, -1, -1
    if '1' in item_lst:
        med_male_chol[i] = 1
    if '2' in item_lst:
        med_male_bp[i] = 1
    if '3' in item_lst:
        med_male_insu[i] = 1

med_male['med_chol'] = med_male_chol
med_male['med_bp'] = med_male_bp
med_male['med_insu'] = med_male_insu
med_male.drop('6177-0.0', axis = 1, inplace = True)

######################################################################################################
####################################      Female     #################################################
######################################################################################################

med_female = mydf.loc[mydf['31-0.0'] == 0][['eid', '6153-0.0']]
med_female['6153-0.0'].fillna('NA', inplace = True)
med_female['6153-0.0'].replace(['[-1]', '[-3]'], 'NA', inplace = True)
med_female_chol = np.zeros((med_female.shape[0])).tolist()
med_female_bp = np.zeros((med_female.shape[0])).tolist()
med_female_insu = np.zeros((med_female.shape[0])).tolist()

for i in range(med_female.shape[0]):
    item_lst = med_female['6153-0.0'].iloc[i]
    if item_lst == 'NA':
        med_female_chol[i], med_female_bp[i], med_female_insu[i] = -1, -1, -1
    if '1' in item_lst:
        med_female_chol[i] = 1
    if '2' in item_lst:
        med_female_bp[i] = 1
    if '3' in item_lst:
        med_female_insu[i] = 1

med_female['med_chol'] = med_female_chol
med_female['med_bp'] = med_female_bp
med_female['med_insu'] = med_female_insu
med_female.drop('6153-0.0', axis = 1, inplace = True)

med_df = pd.concat((med_male, med_female), axis = 0)
mydf = pd.merge(mydf, med_df, how = 'left', on = ['eid'])
mydf.replace(-1, np.nan, inplace = True)

nb_med_df = read_data(['137-0.0'], feature_df, eid_df)
nb_med_df.rename(columns = {'137-0.0':'Nb_Meds'}, inplace = True)
mydf = pd.merge(mydf, nb_med_df, how = 'left', on = ['eid'])

mydf.to_csv(outpath + 'MedicationHistory.csv', index=False)

'''
for i in range(mydf.shape[1]):
    na_prop = np.round(mydf.describe().iloc[0,i]/502409*100,1)
    print((mydf.columns[i], na_prop))

'''

med_female.med_chol.value_counts()
med_female.med_bp.value_counts()
med_female.med_insu.value_counts()
med_male.med_chol.value_counts()
med_male.med_bp.value_counts()
med_male.med_insu.value_counts()

mydf.med_chol.value_counts()
mydf.med_bp.value_counts()
mydf.med_insu.value_counts()
