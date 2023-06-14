

import numpy as np
import pandas as pd
import pickle
from Utility.Evaluation_Utilities import *
from sklearn.metrics import confusion_matrix
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/'
death_df = pd.read_csv(dpath + 'PreprocessedData/TargetData/TargetData.csv', usecols = ['eid', 'X0_y'])
eid_df = pd.read_csv(dpath + 'Eid_info_data.csv', usecols=['eid', 'Age', 'Gender', 'Site_code', 'Region_code'])
mydf = pd.merge(eid_df, death_df, how = 'inner', on = ['eid'])

mydf.sort_values(by = ['Region_code', 'Site_code'], ascending=True, inplace = True)
mydf.Region_code.replace([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                         ['London', 'Wales', 'North-West', 'North-East', 'Yorkshire and Humber',
                          'West Midlands', 'East Midlands', 'South-East', 'South-West', 'Scotland'], inplace = True)

mydf.Age.median()
mydf.Age.quantile(0.25)
mydf.Age.quantile(0.75)
mydf.X0_y.sum()/52006*100

Site_code_lst = np.unique(mydf.Site_code)
Region_code_lst = np.unique(mydf.Region_code)

tmp_region_out = []

for region_code in Region_code_lst:
    tmp_reg_df = mydf.loc[mydf.Region_code == region_code]
    region_nb = int(len(tmp_reg_df))
    region_prop = region_nb / 52006
    region_out = str(region_nb) + ' (' + str(np.round(region_prop*100, 1)) + '%)'
    age_median = int(tmp_reg_df.Age.quantile(0.5))
    age_lqr = int(tmp_reg_df.Age.quantile(0.25))
    age_uqr = int(tmp_reg_df.Age.quantile(0.75))
    age_out = str(age_median) + ' [' + str(age_lqr) + '-' + str(age_uqr) + ']'
    female_nb = int(region_nb-tmp_reg_df.Gender.sum())
    female_prop = female_nb / region_nb
    female_out = str(female_nb) + ' (' + str(np.round(female_prop*100, 1)) + '%)'
    male_nb = int(tmp_reg_df.Gender.sum())
    male_prop = male_nb / region_nb
    male_out = str(male_nb) + ' (' + str(np.round(male_prop * 100, 1)) + '%)'
    death_nb = int(tmp_reg_df.X0_y.sum())
    death_prop = death_nb / region_nb
    death_out = str(death_nb) + ' (' + str(np.round(death_prop * 100, 1)) + '%)'
    tmp_region_out.append([region_code, region_out, age_out, female_out, male_out, death_out])
    print([region_code, region_out, age_out, female_out, male_out, death_out])

outdf_region = pd.DataFrame(tmp_region_out)
outdf_region.columns = ['Region', 'Nb_participants', 'Age', 'Female', 'Male', 'Death']
outdf_region.to_csv(outpath + 'SuppData/sTables/sTable9/Region.csv', index = False)


tmp_site_out = []
for region_code in Site_code_lst:
    tmp_reg_df = mydf.loc[mydf.Site_code == region_code]
    tmp_region = tmp_reg_df.Region_code.iloc[0]
    region_nb = int(len(tmp_reg_df))
    region_prop = region_nb / 52006
    region_out = str(region_nb) + ' (' + str(np.round(region_prop*100, 1)) + '%)'
    age_median = int(tmp_reg_df.Age.quantile(0.5))
    age_lqr = int(tmp_reg_df.Age.quantile(0.25))
    age_uqr = int(tmp_reg_df.Age.quantile(0.75))
    age_out = str(age_median) + ' [' + str(age_lqr) + '-' + str(age_uqr) + ']'
    female_nb = int(region_nb-tmp_reg_df.Gender.sum())
    female_prop = female_nb / region_nb
    female_out = str(female_nb) + ' (' + str(np.round(female_prop*100, 1)) + '%)'
    male_nb = int(tmp_reg_df.Gender.sum())
    male_prop = male_nb / region_nb
    male_out = str(male_nb) + ' (' + str(np.round(male_prop * 100, 1)) + '%)'
    death_nb = int(tmp_reg_df.X0_y.sum())
    death_prop = death_nb / region_nb
    death_out = str(death_nb) + ' (' + str(np.round(death_prop * 100, 1)) + '%)'
    tmp_site_out.append([tmp_region, region_code, region_out, age_out, female_out, male_out, death_out])
    print([tmp_region, region_code, region_out, age_out, female_out, male_out, death_out])


outdf_site = pd.DataFrame(tmp_site_out)
outdf_site.columns = ['Region', 'Site', 'Nb_participants', 'Age', 'Female', 'Male', 'Death']
outdf_site.to_csv(outpath + 'SuppData/sTables/sTable3/Site.csv', index = False)



"London" = 0
"Wales" = 1
"North-West" = 2
"North-East" = 3
"Yorkshire and Humber" = 4
"West Midlands" = 5
"East Midlands" = 6
"South-East" = 7
"South-West" = 8
"Scotland" = 9

mydf['Region_code'].replace([10003, 11001, 11002, 11003, 11004, 11005, 11006, 11007, 11008, 11009, 11010,
                             11011, 11012, 11013, 11014, 11016, 11017, 11018, 11020, 11021, 11022, 11023],
                            [2, 2, 7, 1, 9, 9, 5, 7, 2, 3, 4,
                             8, 0, 6, 4, 2, 3, 0, 0, 5, 1, 1], inplace=True)
