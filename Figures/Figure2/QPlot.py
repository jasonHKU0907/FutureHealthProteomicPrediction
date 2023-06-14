

import numpy as np
import pandas as pd
import pickle
from Utility.Evaluation_Utilities import *
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter as kmf
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'
from lifelines.utils import concordance_index

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/'
target_df_file = dpath + 'PreprocessedData/TargetData/TargetData.csv'
cov_df = pd.read_csv(dpath + 'Eid_info_data.csv', usecols = ['eid', 'Age', 'Gender'])

target_code_df = pd.read_csv(dpath + 'Target_code.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()
target_label_dict = dict(zip(target_code_lst, target_code_df.Disease.tolist()))
fold_id_lst = [ele for ele in range(10)]

for target_code in target_code_lst:
    plotdf = pd.DataFrame()
    for fold_id in fold_id_lst:
        prors_test = pd.read_csv(outpath + 'MLModeling/MLP/ResMLP/' + target_code + '/Test' + str(fold_id) + '_ProRS.csv')
        plotdf = pd.concat([plotdf, prors_test], axis=0)
    plotdf = pd.merge(plotdf, cov_df, how = 'left', on = ['eid'])
    plotdf['Gender'].replace([0, 1], [1, 0], inplace = True)
    plotdf_female = plotdf.loc[plotdf.Gender == 0]
    plotdf_female.reset_index(inplace = True, drop = True)
    plotdf_male = plotdf.loc[plotdf.Gender == 1]
    plotdf_male.reset_index(inplace = True, drop = True)
    ob_female_lst, ob_male_lst = [], []
    ob_female_age_lst, ob_male_age_lst = [], []
    for i in range(100):
        l_cut_female, u_cut_female = plotdf_female.ProRS.quantile(i/100), plotdf_female.ProRS.quantile((i+1)/100)
        tmp_idx_female = plotdf_female.index[(plotdf_female.ProRS>l_cut_female) & (plotdf_female.ProRS<=u_cut_female)]
        ob_female_lst.append(plotdf_female.iloc[tmp_idx_female].Y.sum()/len(tmp_idx_female))
        ob_female_age_lst.append(plotdf_female.iloc[tmp_idx_female].Age.mean())
        l_cut_male, u_cut_male = plotdf_male.ProRS.quantile(i / 100), plotdf_male.ProRS.quantile((i + 1) / 100)
        tmp_idx_male = plotdf_male.index[(plotdf_male.ProRS > l_cut_male) & (plotdf_male.ProRS <= u_cut_male)]
        ob_male_lst.append(plotdf_male.iloc[tmp_idx_male].Y.sum() / len(tmp_idx_male))
        ob_male_age_lst.append(plotdf_male.iloc[tmp_idx_male].Age.mean())
    ob_lst = ob_female_lst + ob_male_lst
    ob_age_lst = ob_female_age_lst + ob_male_age_lst
    pro_q_lst = [i for i in range(100)]*2
    gender_lst = [0]*100 + [1]*100
    scatter_df = pd.DataFrame({'ob_prop': ob_lst, 'pro_q_prop': pro_q_lst, 'gender': gender_lst, 'age': ob_age_lst})
    fig, ax = plt.subplots(figsize=(7.05, 5.35))
    sns.scatterplot(data=scatter_df, x="pro_q_prop", y="ob_prop", hue="gender", size='age', sizes=(25, 175), alpha=0.85, palette=['slateblue', 'red'])
    ax.set_title(target_label_dict[target_code], fontsize=24)
    ax.tick_params(axis='y', labelsize=22)
    ax.get_yaxis().set_ticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.tick_params(axis='x', labelsize=22)
    ax.get_xaxis().set_ticks([0, 25, 50, 75, 100])
    plt.tick_params(axis='both', left=True, top=False, right=False, bottom=True, labelleft=True, labeltop=False, labelright=False, labelbottom=False)
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.legend().set_visible(False)
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    plt.margins(0.05, 0.05)
    fig.tight_layout()
    #plt.savefig(outpath + 'SuppData/sFigures/sFigure3/' + target_code + '.png', bbox_inches='tight',pad_inches = 0.05)


# Remind to customize codes for breast and prostate cancer
