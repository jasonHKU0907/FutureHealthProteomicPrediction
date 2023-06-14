

import glob
import os
import numpy as np
import pandas as pd
import re
from sklearn.impute import KNNImputer

def get_standardization(mydf):
    tmp_df = mydf.copy()
    for col in tmp_df.columns:
        ubd = tmp_df[col].quantile(0.995)
        lbd = tmp_df[col].quantile(0.005)
        tmp_df[col].iloc[tmp_df[col]>ubd] = ubd
        tmp_df[col].iloc[tmp_df[col]<lbd] = lbd
        tmp_df[col] = (tmp_df[col] - lbd) / (ubd - lbd)
    return tmp_df

dpath_raw = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/Covariates/RAW/'
dpath_imp = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/Covariates/Imputed/'
dpath1 = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'

pro_eid_df = pd.read_csv(dpath1 + 'PreprocessedData/ProteomicsData/ProteomicsData.csv', usecols = ['eid'])
region_code_df = pd.read_csv(dpath1 + 'Eid_info_data.csv', usecols = ['eid', 'Region_code'])
info_df = pd.merge(pro_eid_df, region_code_df, how = 'left', on = 'eid')
nb_ind = len(info_df)
region_code_lst = [ele for ele in range(10)]

cov_df0 = pd.read_csv(dpath_raw+'DemographicInfo.csv', usecols=['eid', 'Age', 'Gender', 'Ethnicity', 'TDI', 'Education'])
cov_df1 = pd.read_csv(dpath_raw+'Lifestyle.csv', usecols=['eid', 'ALC_Status', 'SMK_Status', 'HealthySleep', 'RegularPA', 'HealthyDiet', 'SocialContact'])
cov_df2 = pd.read_csv(dpath_raw+'PhysicalMeasurements.csv', usecols=['eid', 'BMI', 'BMR', 'Height', 'Weight', 'WaistCir', 'DBP', 'SBP'])
cov_df3 = pd.read_csv(dpath_raw+'DiseaseHistory.csv', usecols=['eid', 'ChronicIllness', 'DIAB_hist', 'HYPT_hist'])
cov_df4 = pd.read_csv(dpath_raw+'MedicationHistory.csv', usecols=['eid', 'med_chol', 'med_bp', 'med_insu', 'Nb_Meds'])
cov_df5 = pd.read_csv(dpath_raw+'FamilyHistory.csv', usecols=['eid', 'fam_diab', 'fam_hbp', 'fam_dm', 'fam_hd'])
cov_df6 = pd.read_csv(dpath_raw+'Biofluids.csv', usecols=['eid', '30600-0.0', '30690-0.0', '30760-0.0', '30780-0.0', '30870-0.0',
                                                          '30740-0.0', '30750-0.0', '30710-0.0', '30770-0.0', '30830-0.0', '30860-0.0',
                                                          '30610-0.0', '30620-0.0', '30650-0.0', '30730-0.0', '30700-0.0',
                                                          '30720-0.0', '30670-0.0', '30000-0.0', '30010-0.0',
                                                          '30020-0.0', '30030-0.0', '30040-0.0', '30050-0.0', '30080-0.0'])

mydf = pd.merge(info_df, cov_df0, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, cov_df1, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, cov_df2, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, cov_df3, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, cov_df4, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, cov_df5, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, cov_df6, how = 'left', on = ['eid'])

categorical_f_lst = ['Gender', 'Ethnicity', 'Education',
                     'ALC_Status', 'SMK_Status', 'HealthySleep', 'RegularPA', 'HealthyDiet', 'SocialContact',
                     'ChronicIllness', 'DIAB_hist', 'HYPT_hist',
                     'med_chol', 'med_bp', 'med_insu',
                     'fam_diab', 'fam_hbp', 'fam_dm', 'fam_hd']

continuous_f_lst = ['Age', 'TDI',
                    'BMI', 'BMR', 'Height', 'Weight', 'WaistCir', 'DBP', 'SBP',
                    'Nb_Meds',
                    '30600-0.0', '30690-0.0', '30760-0.0', '30780-0.0', '30870-0.0', '30740-0.0', '30750-0.0',
                    '30710-0.0', '30770-0.0', '30830-0.0', '30610-0.0', '30620-0.0', '30650-0.0', '30860-0.0',
                    '30730-0.0', '30700-0.0', '30720-0.0', '30670-0.0', '30000-0.0',
                    '30010-0.0', '30020-0.0', '30030-0.0', '30040-0.0', '30050-0.0', '30080-0.0']

for region_id in region_code_lst:
    for ele in categorical_f_lst:
        train_idx = mydf.index[mydf.Region_code != region_id]
        imp_val = int(mydf.iloc[train_idx][ele].mode())
        test_imp_idx = mydf.index[(mydf.Region_code == region_id) & (mydf[ele].isnull() == True)]
        mydf[ele].iloc[test_imp_idx] = imp_val
    print(region_id)

# Convert all ordinal categorical into [0-1] as KNN requires euclidean distances
mydf['Education'] = mydf['Education']/mydf['Education'].max()
mydf['SocialContact'].replace([0, 1, 2, 3], [0, 0.25, 0.5, 1], inplace = True)
mydf['fam_diab'].replace([0, 1, 2], [0, 0.5, 1], inplace = True)
mydf['fam_hbp'].replace([0, 1, 2], [0, 0.5, 1], inplace = True)
mydf['fam_dm'].replace([0, 1, 2], [0, 0.5, 1], inplace = True)
mydf['fam_hd'].replace([0, 1, 2], [0, 0.5, 1], inplace = True)

# Dummy categorical variables: ethnicity
ethnicity_df = pd.get_dummies(mydf['Ethnicity'])
ethnicity_df.columns = ['Ethnicity_White', 'Ethnicity_Asian', 'Ethnicity_Black', 'Ethnicity_Others']
mydf = pd.concat([mydf, ethnicity_df], axis = 1)
mydf.drop(['Ethnicity'], axis = 1, inplace=True)

# Normalize continuous variables
mydf[continuous_f_lst] = get_standardization(mydf[continuous_f_lst])

mydf_out = pd.DataFrame()

import time
time_start = time.time()

for region_id in region_code_lst:
    tmpdf_train = mydf.loc[mydf.Region_code != region_id].copy()
    tmp_cols = tmpdf_train.columns.tolist()[2:]
    tmpdf_test = mydf.loc[mydf.Region_code == region_id].copy()
    tmpdf_test.reset_index(inplace = True)
    knn_imputer = KNNImputer(n_neighbors=50, weights="uniform")
    knn_imputer.fit(tmpdf_train[tmp_cols])
    tmpdf_out = pd.DataFrame(knn_imputer.transform(tmpdf_test[tmp_cols]))
    tmpdf_test[tmp_cols] = tmpdf_out
    mydf_out = pd.concat([mydf_out, tmpdf_test], axis = 0)
    print('Finish and it total cost ' + str(np.round(time.time() - time_start, 1)) + ' seconds')

mydf_out.drop('index', axis = 1, inplace = True)
mydf_out[['eid', 'Region_code']] = mydf_out[['eid', 'Region_code']].astype(int)
mydf_out.sort_values(by = ['eid'], ascending=True, inplace = True)

mydf_out['Education'] = mydf['Education']*8
mydf_out['SocialContact'].replace([0, 0.25, 0.5, 1], [0, 1, 2, 3], inplace = True)
mydf_out['fam_diab'].replace([0, 0.5, 1], [0, 1, 2], inplace = True)
mydf_out['fam_hbp'].replace([0, 0.5, 1], [0, 1, 2], inplace = True)
mydf_out['fam_dm'].replace([0, 0.5, 1], [0, 1, 2], inplace = True)
mydf_out['fam_hd'].replace([0, 0.5, 1], [0, 1, 2], inplace = True)

mydf_out.rename(columns = {'Age':'DM_AGE', 'Gender':'DM_GENDER', 'TDI':'DM_TDI', 'Education':'DM_EDUC',
                           'ALC_Status':'LS_ALC', 'SMK_Status':'LS_SMK', 'RegularPA':'LS_PA',
                           'HealthyDiet':'LS_DIET', 'SocialContact':'LS_SOC', 'HealthySleep':'LS_SLP',
                           'BMI':'PM_BMI', 'BMR':'PM_BMR', 'Weight':'PM_WT', 'Height':'PM_HT',
                           'WaistCir':'PM_WC', 'DBP':'PM_DBP', 'SBP':'PM_SBP',
                           'ChronicIllness':'DH_ILL', 'DIAB_hist':'DH_DIAB', 'HYPT_hist':'DH_HYPT',
                            'med_chol':'MH_CHOL', 'med_bp':'MH_BP', 'med_insu':'MH_INSU', 'Nb_Meds':'MH_NBMED',
                            'fam_diab':'FH_DIAB', 'fam_hbp':'FH_HBP', 'fam_dm':'FH_DM', 'fam_hd':'FH_HD',
                            '30000-0.0':'BF_BC_WBC', '30010-0.0':'BF_BC_RBC', '30020-0.0':'BF_BC_HB',  '30030-0.0':'BF_BC_HCT',
                            '30040-0.0':'BF_BC_MCV', '30050-0.0':'BF_BC_MCH', '30080-0.0':'BF_BC_PLT', '30610-0.0':'BF_LV_AP',
                            '30620-0.0':'BF_LV_ALT', '30650-0.0':'BF_LV_AST', '30670-0.0':'BF_RE_UREA', '30690-0.0':'BF_LP_TC',
                            '30600-0.0':'BF_RE_ALB', '30860-0.0':'BF_RE_TPRO', '30700-0.0':'BF_RE_CR', '30710-0.0':'BF_IF_CRP', '30720-0.0':'BF_RE_CYSC',
                            '30730-0.0':'BF_LV_GGT', '30740-0.0':'BF_GL_GL', '30750-0.0':'BF_GL_HBA1C', '30760-0.0':'BF_LP_HDLC',
                            '30770-0.0':'BF_EC_IGF1', '30780-0.0':'BF_LP_LDLC', '30830-0.0':'BF_EC_SHBG', '30870-0.0':'BF_LP_TRG',
                            'Ethnicity_White':'DM_ETH_W', 'Ethnicity_Asian':'DM_ETH_A',
                           'Ethnicity_Black':'DM_ETH_B', 'Ethnicity_Others':'DM_ETH_O'}, inplace = True)

mydf_out = mydf_out[['eid', 'Region_code',
                     'DM_AGE', 'DM_GENDER', 'DM_ETH_W', 'DM_ETH_A', 'DM_ETH_B', 'DM_ETH_O', 'DM_TDI', 'DM_EDUC',
                     'LS_ALC', 'LS_SMK', 'LS_PA', 'LS_DIET', 'LS_SOC', 'LS_SLP',
                     'PM_BMI', 'PM_BMR', 'PM_WT', 'PM_HT', 'PM_WC', 'PM_DBP', 'PM_SBP',
                     'DH_ILL', 'DH_DIAB', 'DH_HYPT', 'MH_CHOL', 'MH_BP', 'MH_INSU', 'MH_NBMED',
                     'FH_DIAB', 'FH_HBP', 'FH_DM', 'FH_HD',
                     'BF_LP_TC', 'BF_LP_HDLC', 'BF_LP_LDLC', 'BF_LP_TRG',
                     'BF_GL_GL', 'BF_GL_HBA1C', 'BF_IF_CRP',
                     'BF_EC_IGF1', 'BF_EC_SHBG',
                     'BF_LV_AP', 'BF_LV_ALT', 'BF_LV_AST', 'BF_LV_GGT',
                     'BF_RE_ALB', 'BF_RE_TPRO', 'BF_RE_CR', 'BF_RE_CYSC', 'BF_RE_UREA',
                     'BF_BC_WBC', 'BF_BC_RBC', 'BF_BC_HB', 'BF_BC_HCT', 'BF_BC_MCV', 'BF_BC_MCH', 'BF_BC_PLT']]

mydf_out.to_csv(dpath1 + 'PreprocessedData/Covariates/PANEL.csv', index = False)


for i in range(59):
    print(mydf_out.iloc[:,i].describe()[0])
