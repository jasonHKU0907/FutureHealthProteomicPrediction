

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

target_FieldID = ['20107-0.0', '20110-0.0']
mydf = read_data(target_FieldID, feature_df, eid_df)


'''
derived from 20107-0.0 (father) and 20110-0.0 (mother)
parent_diab: parents history of diabetes, 2: both; 1: single; 0: neither
parent_hbp: parents history of high blood pressure, 2: both; 1: single; 0: neither
parent_dm: parents history of dementia, 2: both; 1: single; 0: neither
parent_str: parents history of stroke, 2: both; 1: single; 0: neither
parent_hd: parents history of heart disease, 2: both; 1: single; 0: neither
parent_bst: mother history of breast cancer, 1: yes; 0: no
parent_pst: father history of Prostate disease, 1: yes; 0: no
'''

######################################################################################################
####################################    Mother     ###################################################
######################################################################################################

mother_hist = mydf[['eid', '20110-0.0']]
mother_hist['20110-0.0'].fillna('NA', inplace = True)
mother_hist['hist'] = mother_hist['20110-0.0']

for i in range(len(mother_hist)):
    item = mother_hist['20110-0.0'].iloc[i]
    if '-11' in item:
        mother_hist['hist'].iloc[i] = 'NA'
    if '-13' in item:
        mother_hist['hist'].iloc[i] = 'NA'

mother_diab = np.zeros(len(mother_hist)).tolist()
mother_hbp = np.zeros(len(mother_hist)).tolist()
mother_dm = np.zeros(len(mother_hist)).tolist()
mother_bst = np.zeros(len(mother_hist)).tolist()
mother_str = np.zeros(len(mother_hist)).tolist()
mother_hd = np.zeros(len(mother_hist)).tolist()

for i in range(len(mother_hist)):
    item_lst = mother_hist['hist'].iloc[i]
    if item_lst == 'NA':
        mother_hbp[i], mother_diab[i], mother_dm[i], mother_bst[i], mother_str[i], mother_hd[i] = -1, -1, -1, -1, -1, -1
    if '8' in item_lst:
        mother_hbp[i] = 1
    if '9' in item_lst:
        mother_diab[i] = 1
    if '10' in item_lst:
        mother_dm[i] = 1
    if '5' in item_lst:
        mother_bst[i] = 1
    if ('[2' in item_lst) | (', 2' in item_lst):
        mother_str[i] = 1
    if ('[1,' in item_lst) | (', 1]' in item_lst) | ('[1]' in item_lst) | (', 1, ' in item_lst):
        mother_hd[i] = 1


mother_hist['mother_diab'] = mother_diab
mother_hist['mother_hbp'] = mother_hbp
mother_hist['mother_dm'] = mother_dm
mother_hist['mother_bst'] = mother_bst
mother_hist['mother_str'] = mother_str
mother_hist['mother_hd'] = mother_hd
mother_hist.drop(['20110-0.0', 'hist'], axis = 1, inplace = True)

######################################################################################################
####################################    Father     ###################################################
######################################################################################################


father_hist = mydf[['eid', '20107-0.0']]
father_hist['20107-0.0'].fillna('NA', inplace = True)
father_hist['hist'] = father_hist['20107-0.0']

for i in range(len(father_hist)):
    item = father_hist['20107-0.0'].iloc[i]
    if '-11' in item:
        father_hist['hist'].iloc[i] = 'NA'
    if '-13' in item:
        father_hist['hist'].iloc[i] = 'NA'

father_diab = np.zeros(len(father_hist)).tolist()
father_hbp = np.zeros(len(father_hist)).tolist()
father_dm = np.zeros(len(father_hist)).tolist()
father_pst = np.zeros(len(father_hist)).tolist()
father_str = np.zeros(len(father_hist)).tolist()
father_hd = np.zeros(len(father_hist)).tolist()

for i in range(len(father_hist)):
    item_lst = father_hist['hist'].iloc[i]
    if item_lst == 'NA':
        father_diab[i], father_hbp[i], father_dm[i], father_pst[i], father_str[i], father_hd[i] = -1, -1, -1, -1, -1, -1
    if '9' in item_lst:
        father_diab[i] = 1
    if '8' in item_lst:
        father_hbp[i] = 1
    if '10' in item_lst:
        father_dm[i] = 1
    if '13' in item_lst:
        father_pst[i] = 1
    if ('[2' in item_lst) | (', 2' in item_lst):
        father_str[i] = 1
    if ('[1,' in item_lst) | (', 1]' in item_lst) | ('[1]' in item_lst) | (', 1, ' in item_lst):
        father_hd[i] = 1

father_hist['father_diab'] = father_diab
father_hist['father_hbp'] = father_hbp
father_hist['father_dm'] = father_dm
father_hist['father_pst'] = father_pst
father_hist['father_str'] = father_str
father_hist['father_hd'] = father_hd
father_hist.drop(['20107-0.0', 'hist'], axis = 1, inplace = True)


######################################################################################################
###########################    Merge Covarites     ###################################################
######################################################################################################


parents_hist = pd.merge(mother_hist, father_hist, how = 'left', on=['eid'])
parents_hist.replace(-1, np.nan, inplace = True)
parents_hist['fam_diab'] = parents_hist['father_diab'] + parents_hist['mother_diab']
parents_hist['fam_hbp'] = parents_hist['father_hbp'] + parents_hist['mother_hbp']
parents_hist['fam_dm'] = parents_hist['father_dm'] + parents_hist['mother_dm']
parents_hist['fam_str'] = parents_hist['father_str'] + parents_hist['mother_str']
parents_hist['fam_hd'] = parents_hist['father_hd'] + parents_hist['mother_hd']
parents_hist['fam_pst'] = parents_hist['father_pst']
parents_hist['fam_bst'] = parents_hist['mother_bst']
mydf = pd.merge(mydf, parents_hist, how = 'left', on = ['eid'])
mydf.to_csv(outpath + 'FamilyHistory.csv', index=False)

mydf.fam_diab.describe()
mydf.fam_diab.value_counts()
mydf.fam_hbp.describe()
mydf.fam_hbp.value_counts()
mydf.fam_dm.describe()
mydf.fam_dm.value_counts()
mydf.fam_str.describe()
mydf.fam_str.value_counts()
mydf.fam_hd.describe()
mydf.fam_hd.value_counts()

'''
for i in range(mydf.shape[1]):
    na_prop = np.round(mydf.describe().iloc[0,i]/502409*100,1)
    print((mydf.columns[i], na_prop))

'''