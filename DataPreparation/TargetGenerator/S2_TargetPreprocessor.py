

import glob
import os
import numpy as np
import pandas as pd
import re

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'

data_files = np.sort(glob.glob(dpath + 'Targets_RAW/*.csv')).tolist()
pro_id_df = pd.read_csv(dpath + 'PreprocessedData/ProteomicsData/ProteomicsData.csv', usecols = ['eid'])
info_df = pd.read_csv(dpath + 'Eid_info_data.csv', usecols=['eid', 'Region_code', 'BL2End_yrs', 'BL2Death_yrs'])
mydf = pd.merge(info_df, pro_id_df, how = 'right', on = ['eid'])

target_f_lst, years_f_lst = [], []

for file in data_files:
    basename = os.path.basename(file)[:2]
    try:
        tmpdf = pd.read_csv(file, usecols=['eid', 'target_y', 'BL2Target_yrs', 'Target_source'])
        tmpdf.columns = ['eid', basename+'_source', basename+'_y', basename+'_years']
        mydf = pd.merge(mydf, tmpdf, how = 'left', on = ['eid'])
    except:
        tmpdf = pd.read_csv(file, usecols=['eid', 'target_y', 'BL2Target_yrs'])
        tmpdf.columns = ['eid', basename+'_y', basename+'_years']
        mydf = pd.merge(mydf, tmpdf, how = 'left', on = ['eid'])

mydf.to_csv(dpath + 'PreprocessedData/TargetData/TargetData.csv', index = False)
