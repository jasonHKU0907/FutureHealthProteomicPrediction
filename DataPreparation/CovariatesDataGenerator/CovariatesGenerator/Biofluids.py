

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

def get_days_intervel(start_date_var, end_date_var, df):
    start_date = pd.to_datetime(df[start_date_var], dayfirst=True)
    end_date = pd.to_datetime(df[end_date_var], dayfirst=True)
    nb_of_dates = start_date.shape[0]
    days = [(end_date[i] - start_date[i]).days for i in range(nb_of_dates)]
    my_yrs = [ele/365 for ele in days]
    return pd.DataFrame(my_yrs)

def get_binary(var_source, df):
    tmp_binary = df[var_source].copy()
    tmp_binary.loc[tmp_binary >= -1] = 1
    tmp_binary.replace(np.nan, 0, inplace=True)
    return tmp_binary


dpath = '/Volumes/JasonWork/Dataset/UKB_Tabular_merged_10/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/Covariates/RAW/'
feature_df = pd.read_csv(dpath + 'UKB_FieldID_Subset.csv')
eid_df = pd.read_csv(dpath + 'UKB_eid.csv')

target_FieldID = ['31-0.0',
                  '30690-0.0', '30760-0.0', '30780-0.0', '30870-0.0',
                  '30740-0.0', '30750-0.0',
                  '30700-0.0', '30720-0.0', '30670-0.0', '30880-0.0',
                  '30600-0.0', '30610-0.0', '30620-0.0', '30650-0.0', '30730-0.0',
                  '30710-0.0',
                  '30010-0.0', '30000-0.0',
                  '30020-0.0', '30030-0.0', '30080-0.0',
                  '30040-0.0', '30050-0.0', '30060-0.0',
                  '30770-0.0', '30830-0.0', '30860-0.0']
mydf = read_data(target_FieldID, feature_df, eid_df)

mydf.sort_values(by = ['eid'], inplace = True)
mydf.to_csv(outpath + 'Biofluids.csv', index=False)


'''
for i in range(mydf.shape[1]):
    na_prop = np.round(mydf.describe().iloc[0,i]/502409*100,1)
    print((mydf.columns[i], na_prop))

[np.round(mydf.describe().iloc[0,i]/502409*100,1) for i in range(mydf.shape[1])]

mydf.columns = ['Gender',
           'Total_CL', 'HDL_CL', 'LDL_CL', 'TG',
           'Glucose', 'HbA1c',
           'Cr', 'CysC', 'UR', 'UA',
           'ALB', 'AP', 'ALT', 'AST', 'GGT',
           'CRP',
           'RBC', 'WBC',
           'Hb', 'Hct', 'PLT',
           'MCV', 'MCH', 'MCHC',
           'IGF1', 'SHBG', 'TPR']
'''
