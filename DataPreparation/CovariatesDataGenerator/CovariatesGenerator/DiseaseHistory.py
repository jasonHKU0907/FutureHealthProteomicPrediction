

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

dpath = '/Volumes/JasonWork/Dataset/UKB_Tabular_merged_10/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/Covariates/RAW/'
feature_df = pd.read_csv(dpath + 'UKB_FieldID_Subset.csv')
eid_df = pd.read_csv(dpath + 'UKB_eid.csv')

diab_f_lst = ['130706-0.0', '130708-0.0', '130710-0.0', '130712-0.0', '130714-0.0']
hypt_f_lst = ['131286-0.0', '131288-0.0', '131290-0.0', '131292-0.0', '131294-0.0', '131294-0.0']
target_FieldID = ['53-0.0', '2188-0.0'] + diab_f_lst + hypt_f_lst
mydf = read_data(target_FieldID, feature_df, eid_df)

mydf[diab_f_lst] = mydf[diab_f_lst].replace(['1900-01-01', '1901-01-01', '1902-02-02', '1903-03-03', '2037-07-07'], '1900-01-01')
tmpdf = mydf[diab_f_lst]
mydf['DIAB_date'] = pd.DataFrame([pd.to_datetime(tmpdf.iloc[i,:], dayfirst=True).min() for i in range(len(tmpdf))])
mydf['BL2DIAB_yrs'] = get_days_intervel('53-0.0', 'DIAB_date', mydf)
mydf['DIAB_hist'] = 0
mydf['DIAB_hist'].loc[mydf['BL2DIAB_yrs']<0] = 1

#mydf[hypt_f_lst] = mydf[hypt_f_lst].replace(['1900-01-01', '1901-01-01', '1902-02-02', '1903-03-03', '2037-07-07'], '1900-01-01')
tmpdf = mydf[hypt_f_lst]
mydf['HYPT_date'] = pd.DataFrame([pd.to_datetime(tmpdf.iloc[i,:], dayfirst=True).min() for i in range(len(tmpdf))])
mydf['BL2HYPT_yrs'] = get_days_intervel('53-0.0', 'HYPT_date', mydf)
mydf['HYPT_hist'] = 0
mydf['HYPT_hist'].loc[mydf['BL2HYPT_yrs']<0] = 1

mydf.rename(columns = {'2188-0.0':'ChronicIllness', '53-0.0':'BL_date'}, inplace = True)
mydf.ChronicIllness.value_counts()
mydf['ChronicIllness'].replace([-1, -3, np.nan], np.nan, inplace = True)

myout = mydf[['BL_date', 'ChronicIllness', 'DIAB_date', 'BL2DIAB_yrs', 'DIAB_hist', 'HYPT_date', 'BL2HYPT_yrs', 'HYPT_hist']]

mydf.to_csv(outpath + 'DiseaseHistory.csv', index=False)

