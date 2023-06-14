

import numpy as np
import pandas as pd
import pickle
from Utility.Evaluation_Utilities import *
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter as kmf
from lifelines import KaplanMeierFitter
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/'

eid_info_df = pd.read_csv(dpath + 'Eid_info_data.csv', usecols = ['eid', 'Gender'])
target_code_df = pd.read_csv(dpath + 'Target_code.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()
fold_id_lst = [ele for ele in range(10)]

def get_output(prop_lst, n=100):
    nb_item = len(prop_lst)
    my_mean = np.mean(prop_lst)
    my_std = np.std(prop_lst)
    my_lbd = my_mean - 1.96*my_std/np.sqrt(nb_item)
    my_ubd = my_mean + 1.96*my_std/np.sqrt(nb_item)
    out_mean = f'{my_mean*n:.2f}' + '%'
    out_lbd = f'{my_lbd*n:.2f}' + '%'
    out_ubd = f'{my_ubd*n:.2f}' + '%'
    out = out_mean + ' [' + out_lbd + '-' + out_ubd + ']'
    return out


my_out_lst = []
for target_code in target_code_lst:
    prop_high_lst, prop_med_lst, prop_low_lst = [], [], []
    or_high2low = []
    for fold_id in fold_id_lst:
        tmpdf = pd.read_csv(outpath + 'MLModeling/MLP/ResMLP/' + target_code + '/Test' + str(fold_id) + '_ProRS.csv')
        tmpdf_high = tmpdf.loc[tmpdf["ProRS"] > tmpdf["ProRS"].quantile(0.667)]
        tmpdf_med = tmpdf.loc[((tmpdf["ProRS"] >= tmpdf["ProRS"].quantile(0.333)) & (tmpdf["ProRS"] <= tmpdf["ProRS"].quantile(0.667)))]
        tmpdf_low = tmpdf.loc[tmpdf["ProRS"] < tmpdf["ProRS"].quantile(0.333)]
        nb_y_all = len(tmpdf)
        prop_high_lst.append(tmpdf_high.Y.sum() / nb_y_all)
        prop_med_lst.append(tmpdf_med.Y.sum() / nb_y_all)
        prop_low_lst.append(tmpdf_low.Y.sum() / nb_y_all)
        or_high2low.append(tmpdf_high.Y.sum()/tmpdf_low.Y.sum())
    prop_high_out = get_output(prop_high_lst, n=100)
    prop_med_out = get_output(prop_med_lst, n=100)
    prop_low_out = get_output(prop_low_lst, n=100)
    or_high2low = np.array(or_high2low)
    or_mean = np.mean(or_high2low[or_high2low!=np.inf])
    or_std = np.std(or_high2low[or_high2low!=np.inf])
    or_lbd = or_mean - 1.96*or_std/np.sqrt(10)
    or_ubd = or_mean + 1.96*or_std/np.sqrt(10)
    or_out = f'{or_mean:.2f}' + ' [' + f'{or_lbd:.2f}' + '-' + f'{or_ubd:.2f}' + ']'
    my_out_lst.append([target_code, prop_high_out, prop_med_out, prop_low_out, or_out])

myout_df = pd.DataFrame(my_out_lst)
myout_df.columns = ['Disease_code', 'Top_tertile', 'Medium_tertile', 'Bottom_tertile', 'OddsRatio']

myout_df = pd.merge(target_code_df, myout_df, how = 'left', on = ['Disease_code'])
myout_df.to_csv(outpath + 'SuppData/sTables/sTable5/sTable5.csv')





#c4 breat cancer & c5 prostate cancer
target_code = 'C4'
eid_info_df = eid_info_df.loc[eid_info_df.Gender == 0]
target_code = 'C5'
eid_info_df = eid_info_df.loc[eid_info_df.Gender == 1]

prop_high_lst, prop_med_lst, prop_low_lst = [], [], []
or_high2low = []
for fold_id in fold_id_lst:
    tmpdf = pd.read_csv(outpath + 'MLModeling/MLP/ResMLP/' + target_code + '/Test' + str(fold_id) + '_ProRS.csv')
    tmpdf = pd.merge(tmpdf, eid_info_df, how = 'inner', on = 'eid')
    tmpdf_high = tmpdf.loc[tmpdf["ProRS"] > tmpdf["ProRS"].quantile(0.667)]
    tmpdf_med = tmpdf.loc[((tmpdf["ProRS"] >= tmpdf["ProRS"].quantile(0.333)) & (tmpdf["ProRS"] <= tmpdf["ProRS"].quantile(0.667)))]
    tmpdf_low = tmpdf.loc[tmpdf["ProRS"] < tmpdf["ProRS"].quantile(0.333)]
    nb_y_all = len(tmpdf)
    prop_high_lst.append(tmpdf_high.Y.sum() / nb_y_all)
    prop_med_lst.append(tmpdf_med.Y.sum() / nb_y_all)
    prop_low_lst.append(tmpdf_low.Y.sum() / nb_y_all)
    or_high2low.append(tmpdf_high.Y.sum() / tmpdf_low.Y.sum())
prop_high_out = get_output(prop_high_lst, n=100)
prop_med_out = get_output(prop_med_lst, n=100)
prop_low_out = get_output(prop_low_lst, n=100)
or_high2low = np.array(or_high2low)
or_mean = np.mean(or_high2low[or_high2low != np.inf])
or_std = np.std(or_high2low[or_high2low != np.inf])
or_lbd = or_mean - 1.96 * or_std / np.sqrt(10)
or_ubd = or_mean + 1.96 * or_std / np.sqrt(10)
or_out = f'{or_mean:.2f}' + ' [' + f'{or_lbd:.2f}' + '-' + f'{or_ubd:.2f}' + ']'
print(target_code)
print(prop_high_out)
print(prop_med_out)
print(prop_low_out)
print(or_out)





