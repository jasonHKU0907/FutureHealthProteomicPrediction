

import numpy as np
import pandas as pd
import pickle
from Utility.Evaluation_Utilities import *
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter as kmf
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
from lifelines.utils import concordance_index

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/'
target_df_file = dpath + 'PreprocessedData/TargetData/TargetData.csv'
cov_df = pd.read_csv(dpath + 'PreprocessedData/Covariates/PANEL.csv', usecols = ['eid', 'DM_AGE', 'DM_GENDER'])

target_code_df = pd.read_csv(dpath + 'Target_code.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()
target_label_dict = dict(zip(target_code_lst, target_code_df.Disease.tolist()))
fold_id_lst = [ele for ele in range(10)]


for target_code in target_code_lst:
    plotdf = pd.DataFrame()
    for fold_id in fold_id_lst:
        prors_test = pd.read_csv(outpath + 'MLModeling/MLP/ResMLP/' + target_code + '/Test' + str(fold_id) + '_ProRS.csv')
        prors_test['ProRS'] = prors_test['ProRS'] * 100
        plotdf = pd.concat([plotdf, prors_test], axis=0)
    plotdf.reset_index(inplace=True, drop=True)
    high = (plotdf["ProRS"] > plotdf["ProRS"].quantile(0.667))
    medium = ((plotdf["ProRS"] >= plotdf["ProRS"].quantile(0.333)) & (plotdf["ProRS"] <= plotdf["ProRS"].quantile(0.667)))
    low = (plotdf["ProRS"] < plotdf["ProRS"].quantile(0.333))
    fig, ax = plt.subplots(figsize=(8, 6))
    #ax.set_facecolor("whitesmoke")
    kmf = KaplanMeierFitter()
    kmf.fit(durations=plotdf.Y_yrs[high], event_observed=plotdf.Y[high], label='High')
    kmf.plot_survival_function(ax=ax, color='darkred', linewidth=3)
    kmf.fit(durations=plotdf.Y_yrs[medium], event_observed=plotdf.Y[medium], label='Medium')
    kmf.plot_survival_function(ax=ax, color='orange', linewidth=3)
    kmf.fit(durations=plotdf.Y_yrs[low], event_observed=plotdf.Y[low], label="Low")
    kmf.plot_survival_function(ax=ax, color='deepskyblue', linewidth=3)
    ax.set_title(target_label_dict[target_code], fontsize=24)
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=22)
    ax.get_xaxis().set_ticks([0, 5, 10, 15])
    #ax.get_yaxis().set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.get_yaxis().set_ticks(my_xais_lst)
    plt.tick_params(axis='both', left=True, top=False, right=False, bottom=True, labelleft=True, labeltop=False,
                    labelright=False, labelbottom=False)
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.legend().set_visible(False)
    ax.xaxis.label.set_visible(False)
    plt.margins(0.0, 0.05)
    plt.savefig(outpath + 'SuppData/sFigures/sFigure2/' + target_code + '.png', bbox_inches='tight',pad_inches = 0.05)

# Remind to customize codes for breast and prostate cancer
