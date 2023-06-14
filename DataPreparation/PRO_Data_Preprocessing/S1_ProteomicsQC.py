
# This file simply try to find the proportion of missingness in each individuals
# Individuals with Prop of NA>30% will be removed for further modeling and statistical analysis

import glob
import os
import numpy as np
import pandas as pd
import re

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
pro_df = pd.read_csv(dpath + 'Proteomics_RAW/Proteomics_ins0.csv')

my_eid_lst = pro_df.eid.tolist()
nb_eids = len(my_eid_lst)
my_pros_lst = pro_df.columns.tolist()[:-1]


na_prop_lst = []
for my_pros in my_pros_lst:
    tmpdf = pro_df[my_pros]
    nb_na = len(np.where(tmpdf.isna() == True)[0])
    na_prop_lst.append(np.round(nb_na/nb_eids, 3))

na_pros_df = pd.DataFrame({'Pros': my_pros_lst, 'NA_prop':na_prop_lst})
na_pros_df.sort_values(ascending=False, by = ['NA_prop'], inplace = True)
pros_beyond_na30_lst = na_pros_df.loc[na_pros_df.NA_prop>0.3].Pros.tolist()
pro_df.drop(pros_beyond_na30_lst, axis = 1, inplace=True)


nb_pros = pro_df.shape[1] -1

na_prop_pros_lst = []
for my_eid in my_eid_lst:
    tmpdf = pro_df.loc[pro_df.eid == my_eid]
    nb_na = len(np.where(tmpdf.isna() == True)[0])
    na_prop_pros_lst.append(np.round(nb_na/nb_pros, 3))

na_eid_df = pd.DataFrame({'eid': my_eid_lst, 'NA_prop':na_prop_pros_lst})
na_eid_df.sort_values(ascending=False, by = ['NA_prop'], inplace = True)

eid_beyond_na30_lst = na_eid_df.loc[na_eid_df.NA_prop>0.3].eid.tolist()

pro_df = pro_df[~pro_df.eid.isin(eid_beyond_na30_lst)]

pro_id_lst = pro_df.columns[:-1].tolist()
pro_df_preprocessed = pro_df[['eid'] + pro_id_lst]
pro_df_preprocessed.to_csv(dpath + 'Proteomics_RAW/Proteomics_S1QC.csv', index = False)


