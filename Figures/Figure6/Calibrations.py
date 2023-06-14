
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/'

target_code_df = pd.read_csv(dpath + 'Target_code.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()
target_label_dict = dict(zip(target_code_lst, target_code_df.Disease.tolist()))

for target_code in target_code_lst:
    nb_bins = 10
    mydf0 = pd.read_csv(outpath + 'MLModeling/CPH/PredProbs/ProRS/' + target_code + '.csv')
    mydf1 = pd.read_csv(outpath + 'MLModeling/CPH/PredProbs/AgeSex/' + target_code + '.csv')
    mydf2 = pd.read_csv(outpath + 'MLModeling/CPH/PredProbs/PANEL/' + target_code + '.csv')
    mydf10 = pd.read_csv(outpath + 'MLModeling/CPH/PredProbs/ProRS_AgeSex/' + target_code + '.csv')
    mydf20 = pd.read_csv(outpath + 'MLModeling/CPH/PredProbs/ProRS_PANEL/' + target_code + '.csv')
    fold_id_lst = [ele for ele in range(10)]
    prob_true0, prob_pred0 = calibration_curve(mydf0.Y, mydf0.ProRS, n_bins=nb_bins, strategy='quantile')
    prob_true1, prob_pred1 = calibration_curve(mydf1.Y, mydf1.CPH_Risk, n_bins=nb_bins, strategy='quantile')
    prob_true2, prob_pred2 = calibration_curve(mydf2.Y, mydf2.CPH_Risk, n_bins=nb_bins, strategy='quantile')
    prob_true10, prob_pred10 = calibration_curve(mydf10.Y, mydf10.CPH_Risk, n_bins=nb_bins, strategy='quantile')
    prob_true20, prob_pred20 = calibration_curve(mydf20.Y, mydf20.CPH_Risk, n_bins=nb_bins, strategy='quantile')
    all_probs = prob_true0.tolist() + prob_true1.tolist() + prob_true2.tolist() + prob_true10.tolist() + \
                prob_true20.tolist() + prob_pred0.tolist() + prob_pred1.tolist() + prob_pred2.tolist() + \
                prob_pred10.tolist() + prob_pred20.tolist()
    lim_ubd = np.round(np.max(all_probs) + 0.015, 2)
    fig, ax = plt.subplots(figsize=(12, 12))
    plt.plot([0, lim_ubd], [0, lim_ubd], linestyle='solid', color='lime', linewidth=7)
    plt.xlim([-0.0, lim_ubd])
    plt.ylim([-0.0, lim_ubd])
    plt.plot(prob_pred1, prob_true1, linestyle='solid', color='lightslategray', linewidth=5, label='Model1')
    plt.plot(prob_pred2, prob_true2, linestyle='dashed', linewidth=5, color='darkslategrey', label='Model2')
    plt.plot(prob_pred0, prob_true0, linestyle='solid', linewidth=5, color='firebrick', label='Model0')
    plt.plot(prob_pred10, prob_true10, linestyle='dotted', linewidth=5, color='darkslategrey', label='Model10')
    plt.plot(prob_pred20, prob_true20, linestyle='solid', linewidth=5, color='darkslategrey', label='Model20')
    ax.set_title(target_label_dict[target_code], y=1.0, pad=-36, fontsize=42)
    my_lbs = ax.get_xticks().tolist()
    my_lbs1 = [str(int(ele * 100)) for ele in my_lbs]
    ax.set_xticklabels(my_lbs1, fontsize=30)
    ax.set_yticklabels(my_lbs1, fontsize=30)
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.legend().set_visible(False)
    plt.margins(0.0, 0.05)
    plt.tight_layout()
    plt.savefig(outpath + 'SuppData/sFigures/sFigure4/' + target_code + '.png', bbox_inches='tight', pad_inches=0.05)

