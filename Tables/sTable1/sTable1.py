

import glob
import os
import numpy as np
import pandas as pd
import re

def preprocess_target_individuals(target_df_file, target_code):
    target_y0, target_yrs0, target_source0 = target_code[0] + '0_y', target_code[0] + '0_years', target_code[0] + '0_source'
    target_y, target_yrs, target_source = target_code + '_y', target_code + '_years', target_code + '_source'
    target_df = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y, target_yrs])
    if (('C' in target_code) | ('X' in target_code)):
        target_df0 = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y0, target_yrs0])
        rm_idx0 = target_df0.index[target_df0[target_yrs0] <= 0]
        rm_eid_lst = target_df0.iloc[rm_idx0].eid.tolist()
    else:
        target_df0 = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y0, target_yrs0, target_source0])
        rm_idx0 = target_df0.index[(target_df0[target_source0] >= 50) | (target_df0[target_yrs0] <= 0)]
        rm_eid_lst = target_df0.iloc[rm_idx0].eid.tolist()
    clean_df = target_df[~target_df.eid.isin(rm_eid_lst)]
    clean_df.reset_index(inplace=True, drop = True)
    return clean_df

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/SuppData/sTables/sTable1/'
target_df_file = dpath + 'PreprocessedData/TargetData/TargetData.csv'
cov_df = pd.read_csv(dpath + 'PreprocessedData/ProteomicsData/ProteomicsData_imp.csv', usecols = ['eid', '2912'])
target_code_df = pd.read_csv(dpath + 'Target_code_plot.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()
info_df = pd.read_csv(dpath + 'Eid_info_data.csv', usecols=['eid', 'Gender'])


out_lst = []
for target_code in target_code_lst:
    target_y, target_yrs = target_code + '_y', target_code + '_years'
    target_df = preprocess_target_individuals(target_df_file, target_code)
    #target_df = pd.merge(target_df, info_df, how = 'left', on = ['eid'])
    #target_df = target_df.loc[target_df.Gender == 1]
    nb_exclude = len(cov_df) - len(target_df)
    nb2analysis = len(target_df)
    nb_tgs = target_df[target_y].sum()
    prop_tgs = str(np.round(nb_tgs/nb2analysis*100,2)) + '%'
    out_lst.append([target_code, nb_exclude, nb2analysis, nb_tgs, prop_tgs])

outdf = pd.DataFrame(out_lst)
outdf.columns = ['Disease_code', 'Excluded.participants', 'Available.participants', 'Target.outcomes.after.baseline', 'Proportion.of.target.outcomes']
outdf = pd.merge(outdf, target_code_df, how = 'inner', on = ['Disease_code'])
outdf.to_csv(outpath + 'sTable1.csv')





'''
for H3
def preprocess_target_individuals(target_df_file, target_code):
    target_y0, target_yrs0, target_source0 = target_code[0] + '0_y', target_code[0] + '0_years', target_code[0] + '0_source'
    target_y, target_yrs, target_source = target_code + '_y', target_code + '_years', target_code + '_source'
    target_df = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y, target_yrs, target_source])
    if (('C' in target_code) | ('X' in target_code)):
        target_df0 = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y0, target_yrs0])
        rm_idx0 = target_df0.index[target_df0[target_yrs0] <= 0]
        rm_eid_lst = target_df0.iloc[rm_idx0].eid.tolist()
    else:
        #target_df0 = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y0, target_yrs0, target_source0])
        rm_idx = target_df.index[(target_df[target_source] >= 50) | (target_df[target_yrs] <= 0)]
        rm_eid_lst = target_df.iloc[rm_idx].eid.tolist()
    clean_df = target_df[~target_df.eid.isin(rm_eid_lst)]
    clean_df.reset_index(inplace=True)
    clean_df.drop('index', axis=1, inplace=True)
    return clean_df
'''

'''
for F1
def preprocess_target_individuals(target_df_file, target_code):
    target_y0, target_yrs0, target_source0 = target_code[0] + '0_y', target_code[0] + '0_years', target_code[0] + '0_source'
    target_y, target_yrs, target_source = target_code + '_y', target_code + '_years', target_code + '_source'
    target_df = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y, target_yrs])
    if (('C' in target_code) | ('X' in target_code)):
        target_df0 = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y0, target_yrs0])
        rm_idx0 = target_df0.index[target_df0[target_yrs0] <= 0]
        rm_eid_lst = target_df0.iloc[rm_idx0].eid.tolist()
    else:
        target_df0 = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y0, target_yrs0, target_source0])
        rm_idx0 = target_df0.index[(target_df0[target_source0] >= 50) | (target_df0[target_yrs0] <= 0)]
        rm_eid_lst = target_df0.iloc[rm_idx0].eid.tolist()
    clean_df = target_df[~target_df.eid.isin(rm_eid_lst)]
    rm_idx = clean_df.index[clean_df[target_yrs] <= 0]
    clean_df.drop(rm_idx, axis = 0, inplace = True)
    clean_df.reset_index(inplace=True, drop = True)
    return clean_df
'''
