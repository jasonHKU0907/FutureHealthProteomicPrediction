
'''
This convert original downloaded Olink_data to prepared format of row*columns = eid*pro_ids
'''

import glob
import os
import numpy as np
import pandas as pd
import re

dpath = '/Volumes/JasonWork/Dataset/UKB_Proteomics_20230417/'

# we only extract data obtained at instance 0
mydf = pd.read_csv(dpath + 'olink_data.csv')
mydf = mydf.loc[mydf.ins_index == 0]
mydf.ins_index.value_counts()
mydf.reset_index(inplace = True)
mydf.drop(['index', 'ins_index'], axis = 1, inplace = True)

# find all unique eid and pro_ids
eid_lst = list(np.sort(list(set(mydf.eid))))
protein_lst = list(np.sort(list(set(mydf.protein_id))))
myout_df_final = pd.DataFrame()
step = 5000

# Split the loop into 11 rounds, as pd.concat would slow down with accumulated iterations
# Under each round, we exract all protein data and merge them to the full protein_ids_lst into dataframe
# Then, use pd.concat to sequentially add each eid one by one


for i in range(11):
    protein_tmp_df = pd.DataFrame({'protein_id': protein_lst})
    myout_df = pd.DataFrame({'protein_id': protein_lst})
    j = 0
    for eid_item in eid_lst[i*step:(i+1)*step]:
        tmpdf0 = mydf.loc[mydf.eid == eid_item]
        tmpdf1 = tmpdf0[['protein_id', 'result']]
        merged_df = pd.merge(protein_tmp_df, tmpdf1, how='left', on=['protein_id'])
        myout_df = pd.concat([myout_df, merged_df.iloc[:, 1:2]], axis=1)
        myout_df.rename(columns={'result': eid_item}, inplace=True)
        print(j)
        j+=1
    myout_df = myout_df.transpose()
    myout_df.columns = myout_df.iloc[0].astype(int).tolist()
    myout_df.drop('protein_id', axis=0, inplace=True)
    myout_df['eid'] = myout_df.index.astype(int).tolist()
    myout_df_final = pd.concat([myout_df_final, myout_df], axis = 0)
    myout_df_final.to_csv(dpath + 'Protemics_ins0.csv', index = False)
    print(i)


