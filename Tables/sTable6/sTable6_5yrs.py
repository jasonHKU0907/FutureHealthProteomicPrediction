

import numpy as np
import pandas as pd
import pickle
from Utility.Evaluation_Utilities import *
from sklearn.metrics import brier_score_loss, recall_score, roc_auc_score, average_precision_score
from Utility.Training_Utilities import *
from lifelines.utils import concordance_index
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/'
outpath1 = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/SuppData/sTables/sTable6/Data/5YEARS/'
target_code_df = pd.read_csv(dpath + 'Target_code.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()
fold_id_lst = [ele for ele in range(10)]

def get_eval(y_test, pred_prob, cutoff):
    pred_binary = threshold(pred_prob, cutoff)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_binary).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp)
    Youden = sens + spec - 1
    f1 = 2 * prec * sens / (prec + sens)
    auc = roc_auc_score(y_test, pred_prob)
    apr = average_precision_score(y_test, pred_prob)
    brier = brier_score_loss(y_test, pred_prob)
    nnd = 1 / Youden
    evaluations = np.round((cutoff, auc, acc, sens, spec, prec, Youden, f1, apr, nnd, brier), 4)
    evaluations = pd.DataFrame(evaluations).T
    evaluations.columns = ['Cutoff', 'AUC', 'Acc', 'Sens', 'Spec', 'Prec', 'Youden', 'F1', 'APR', 'NND', 'BRIER']
    return evaluations


for target_code in target_code_lst:
    myout_df = pd.DataFrame()
    for fold_id in fold_id_lst:
        tmpdf = pd.read_csv(outpath + 'MLModeling/MLP/ResMLP/' + target_code + '/Test' + str(fold_id) + '_ProRS.csv')
        tmpdf['Y'].loc[tmpdf.Y_yrs>5] = 0
        tmp_YI_cv = [Youden_index(tmpdf.Y, threshold(tmpdf.ProRS, i * 0.001)) for i in range(0, 1000)]
        opt_ct_idx = tmp_YI_cv.index(np.max(tmp_YI_cv))
        opt_ct = 0.001 * (opt_ct_idx + 1)
        tmpout_df = get_eval(tmpdf.Y, tmpdf.ProRS, opt_ct)
        myout_df = pd.concat([myout_df, tmpout_df], axis = 0)
    myout_df = myout_df.T
    myout_df['MEAN'] = myout_df.mean(axis = 1)
    myout_df['STD'] = myout_df.std(axis = 1)
    myout_df['LBD'] = myout_df['MEAN'] - 1.96*myout_df['STD']/np.sqrt(10)
    myout_df['UBD'] = myout_df['MEAN'] + 1.96*myout_df['STD']/np.sqrt(10)
    my_out_lst = []
    for i in range(11):
        my_mean = f'{myout_df.MEAN[i]:.2f}'
        my_lbd = f'{myout_df.LBD[i]:.2f}'
        my_ubd = f'{myout_df.UBD[i]:.2f}'
        my_out_lst.append(my_mean + ' [' + my_lbd + '-' + my_ubd + ']')
    myout_df['output'] = my_out_lst
    myout_df = myout_df.T
    myout_df.to_csv(outpath1 + target_code + '.csv', index = True)

