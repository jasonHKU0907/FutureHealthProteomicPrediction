

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

target_FieldID = ['1289-0.0', '1299-0.0', '1309-0.0', '1319-0.0', '1329-0.0', '1339-0.0',
                  '1349-0.0', '1369-0.0', '1379-0.0', '1389-0.0',
                  '1438-0.0', '1448-0.0', '1458-0.0', '1468-0.0']
mydf = read_data(target_FieldID, feature_df, eid_df)

######################################################################################################
####################################   Health Diet  ###################################################
######################################################################################################

'''
Vegetable Intake: 1289-0.0 (Cooked vegetable intake) and 1299-0.0 (Salad / raw vegetable intake)
Fruit Intake: 1309-0.0 (Fresh fruit intake) and 1319-0.0 (Dried fruit intake)

At least 4 of the following 7 food groups:
1. Vegetables: ≥ 3 servings/day (1289, 1299)
2. Fruits: ≥ 3 servings/day (1309, 1319)
3. Fish: ≥2 servings/week (1329, 1339)
4. Processed meats: ≤ 1 serving/week (1349)
5. Unprocessed red meats: ≤ 1.5 servings/week (1369, 1379, 1389)
6. Whole grains: ≥ 3servings/day (1438, 1448)
7. Refined grains: ≤1.5servings/day (1458, 1468)

'''

####################################################################################################
##### Vegetables: ≥ 3 servings/day                 #################################################
##### Cooked vegetable intake (Field ID 1289)      #################################################
##### Salad / raw vegetable intake (Field ID 1299) #################################################
####################################################################################################

mydf['1289-0.0'].replace([-10, -1, -3, np.nan], np.nan, inplace = True)
mydf['1299-0.0'].replace([-10, -1, -3, np.nan], np.nan, inplace = True)
veg_intake = mydf['1289-0.0'] + mydf['1299-0.0']
mydf['Veg_intake'] = 0
mydf['Veg_intake'].iloc[veg_intake >= 3] = 1


####################################################################################################
##### Fruits: ≥ 3 servings/day                     #################################################
##### Fresh fruit intake (Field ID 1309)           #################################################
##### Dired fruit intake (Field ID 1319)           #################################################
####################################################################################################

mydf['1309-0.0'].replace([-10, -1, -3, np.nan], int(mydf['1309-0.0'].mode()), inplace = True)
mydf['1319-0.0'].replace([-10, -1, -3, np.nan], int(mydf['1319-0.0'].mode()), inplace = True)
fru_intake = mydf['1309-0.0'] + mydf['1319-0.0']
mydf['Fruit_intake'] = 0
mydf['Fruit_intake'].iloc[fru_intake >= 3] = 1


####################################################################################################
##### Fish: ≥2 servings/week                       #################################################
##### Fresh fruit intake (Field ID 1309)           #################################################
##### Dired fruit intake (Field ID 1319)           #################################################
####################################################################################################

mydf['1329-0.0'].replace([-10, -1, -3, np.nan], int(mydf['1329-0.0'].mode()), inplace = True)
mydf['1339-0.0'].replace([-10, -1, -3, np.nan], int(mydf['1339-0.0'].mode()), inplace = True)

'''
pd.crosstab(mydf['1329-0.0'], mydf['1339-0.0'])
21513 + 17865 + 14778 + 2614 + 75247 + 70657 + 2048 + 37634
242356
'''

fish_intake = []

for i in range(len(mydf)):
    if (mydf['1329-0.0'].iloc[i] >= 3) | (mydf['1339-0.0'].iloc[i] >= 3):
        fish_intake.append(1)
    elif (mydf['1329-0.0'].iloc[i] == 2) & (mydf['1339-0.0'].iloc[i] == 2):
        fish_intake.append(1)
    else:
        fish_intake.append(0)

mydf['Fish_intake'] = pd.DataFrame(fish_intake)

####################################################################################################
##### Processed meats: ≤ 1 serving/week          ###################################################
##### Processed meat intake (Field ID 1349)      ###################################################
####################################################################################################

mydf['Pro_meat'] = 0
mydf['1349-0.0'].replace([-10, -1, -3, np.nan], int(mydf['1349-0.0'].mode()), inplace = True)
mydf['Pro_meat'].loc[mydf['1349-0.0']<=2] = 1

####################################################################################################
##### Unprocessed red meats: ≤ 1.5 servings/week ###################################################
##### Beef intake (Field ID 1369)                ###################################################
##### Lamb/mutton intake (Field ID 1379)         ###################################################
##### Pork intake (Field ID 1389)                ###################################################
####################################################################################################
mydf['1369-0.0'].replace([-10, -1, -3, np.nan], int(mydf['1369-0.0'].mode()), inplace = True)
mydf['1379-0.0'].replace([-10, -1, -3, np.nan], int(mydf['1379-0.0'].mode()), inplace = True)
mydf['1389-0.0'].replace([-10, -1, -3, np.nan], int(mydf['1389-0.0'].mode()), inplace = True)

unPro_RedMeat_intake = []
for i in range(len(mydf)):
    if (mydf['1369-0.0'].iloc[i] <= 1) & (mydf['1379-0.0'].iloc[i] <= 1) & (mydf['1389-0.0'].iloc[i] <= 1):
        unPro_RedMeat_intake.append(1)
    elif (mydf['1329-0.0'].iloc[i] == 2) & (mydf['1339-0.0'].iloc[i] <= 1) & (mydf['1389-0.0'].iloc[i] <= 1):
        unPro_RedMeat_intake.append(2)
    elif (mydf['1329-0.0'].iloc[i] <= 1) & (mydf['1339-0.0'].iloc[i] == 2) & (mydf['1389-0.0'].iloc[i] <= 1):
        unPro_RedMeat_intake.append(2)
    elif (mydf['1329-0.0'].iloc[i] <= 1) & (mydf['1339-0.0'].iloc[i] <= 1) & (mydf['1389-0.0'].iloc[i] == 2):
        unPro_RedMeat_intake.append(2)
    else:
        unPro_RedMeat_intake.append(0)

mydf['unPro_RedMeat'] = pd.DataFrame(unPro_RedMeat_intake)

####################################################################################################
##### Whole grains: ≥ 3servings/day (1438, 1448) ###################################################
##### Bread intake (Field ID 1438)               ###################################################
##### Bread type  (Field ID 1448)                ###################################################
####################################################################################################

mydf['1438-0.0'].replace([-10, -1, -3, np.nan], int(mydf['1438-0.0'].mode()), inplace = True)
mydf['1448-0.0'].replace([-10, -1, -3, np.nan], int(mydf['1448-0.0'].mode()), inplace = True)

mydf['Who_grain'] = 0
mydf['Who_grain'].loc[((mydf['1438-0.0']>=3) & (mydf['1448-0.0']==3))] = 1


####################################################################################################
##### Refined grains: ≤1.5servings/day           ###################################################
##### Cereal intake (Field ID 1458)              ###################################################
##### Cereal type  (Field ID 1468)               ###################################################
####################################################################################################

mydf['1458-0.0'].replace([-10, -1, -3, np.nan], int(mydf['1458-0.0'].mode()), inplace = True)
mydf['1468-0.0'].replace([-10, -1, -3, np.nan], int(mydf['1468-0.0'].mode()), inplace = True)
mydf['Ref_grain'] = 0
mydf['Ref_grain'].loc[mydf['1458-0.0']<=1.5] = 1

###################################################################################
##### Get final score           ###################################################
###################################################################################

mydf['HealthyDiet_raw'] = mydf['Veg_intake'] + mydf['Fruit_intake'] + mydf['Fish_intake'] + mydf['Pro_meat'] + \
                          mydf['unPro_RedMeat'] + mydf['Who_grain'] + mydf['Ref_grain']
mydf['HealthyDiet_raw'].value_counts()
mydf['HealthyDiet'] = mydf['HealthyDiet_raw'].copy()
mydf['HealthyDiet'].loc[mydf['HealthyDiet'] <4] = 0
mydf['HealthyDiet'].loc[mydf['HealthyDiet'] >=4] = 1
mydf['HealthyDiet'].value_counts()

mydf.to_csv(outpath + '/Lifestyle_Diet.csv', index=False)

