
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all


def get_xlim(net_benefit):
    idx = 0
    while np.sum(net_benefit[idx:(idx+10)])  >= 0.0001:
        idx+=1
    return idx*0.001


dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/'

target_code_df = pd.read_csv(dpath + 'Target_code.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()
target_label_dict = dict(zip(target_code_lst, target_code_df.Disease.tolist()))


for target_code in target_code_lst:
    xlim_ubd = 1
    mydf0 = pd.read_csv(outpath + 'MLModeling/CPH/PredProbs/ProRS/' + target_code + '.csv')
    mydf1 = pd.read_csv(outpath + 'MLModeling/CPH/PredProbs/AgeSex/' + target_code + '.csv')
    mydf2 = pd.read_csv(outpath + 'MLModeling/CPH/PredProbs/PANEL/' + target_code + '.csv')
    mydf10 = pd.read_csv(outpath + 'MLModeling/CPH/PredProbs/ProRS_AgeSex/' + target_code + '.csv')
    mydf20 = pd.read_csv(outpath + 'MLModeling/CPH/PredProbs/ProRS_PANEL/' + target_code + '.csv')
    fold_id_lst = [ele for ele in range(10)]
    fig, ax = plt.subplots(figsize=(12, 12))
    thresh_group = np.arange(0, xlim_ubd, 0.001)
    net_benefit_all = calculate_net_benefit_all(thresh_group, mydf1.Y)
    y_all = np.maximum(net_benefit_all, 0)
    ax.plot(thresh_group, net_benefit_all, linewidth=5, color='dimgrey', label='Treat all')
    ax.set_xlim(0, xlim_ubd)
    ax.set_ylim(0, net_benefit_all.max())
    net_benefit_model1 = calculate_net_benefit_model(thresh_group, mydf1.CPH_Risk, mydf1.Y)
    ax.plot(thresh_group, net_benefit_model1, linestyle='solid', color='lightslategray', linewidth=3.5, label='Model1')
    y1 = np.maximum(net_benefit_model1, 0)
    net_benefit_model2 = calculate_net_benefit_model(thresh_group, mydf2.CPH_Risk, mydf2.Y)
    ax.plot(thresh_group, net_benefit_model2, linestyle='dashed', linewidth=3.5, color='darkslategrey', label='Model2')
    y2 = np.maximum(net_benefit_model2, y_all)
    ax.fill_between(thresh_group, y1, y2, color='red', alpha=0.15)
    net_benefit_model0 = calculate_net_benefit_model(thresh_group, mydf0.ProRS, mydf0.Y)
    ax.plot(thresh_group, net_benefit_model0, linestyle='solid', linewidth=3.5, color='firebrick', label='Model0')
    y0 = np.maximum(net_benefit_model0, y2)
    ax.fill_between(thresh_group, y0, y2, color='red', alpha=0.15)
    y0 = np.maximum(net_benefit_model0, y1)
    ax.fill_between(thresh_group, y0, y1, color='red', alpha=0.15)
    net_benefit_model10 = calculate_net_benefit_model(thresh_group, mydf10.CPH_Risk, mydf10.Y)
    ax.plot(thresh_group, net_benefit_model10, linestyle='dotted', linewidth=3.5, color='darkslategrey', label='Model10')
    y10 = np.maximum(net_benefit_model10, y0)
    # ax.fill_between(thresh_group, y10, y0, color='purple', alpha=0.2)
    net_benefit_model20 = calculate_net_benefit_model(thresh_group, mydf20.CPH_Risk, mydf20.Y)
    ax.plot(thresh_group, net_benefit_model20, linestyle='solid', linewidth=3.5, color='darkslategrey', label='Model20')
    y20 = np.maximum(net_benefit_model20, y0)
    ax.fill_between(thresh_group, y20, y0, color='lightseagreen', alpha=0.25)
    x_lim1 = get_xlim(net_benefit_model1)
    x_lim2 = get_xlim(net_benefit_model2)
    x_lim0 = get_xlim(net_benefit_model0)
    x_lim10 = get_xlim(net_benefit_model10)
    x_lim20 = get_xlim(net_benefit_model20)
    xlim_ubd_new = np.max((x_lim1, x_lim2, x_lim0, x_lim10, x_lim20))
    if xlim_ubd_new>=0.35:
        xlim_ubd_new = xlim_ubd_new+0.05
    elif (xlim_ubd_new <0.35) & (xlim_ubd_new>0.15):
        xlim_ubd_new = xlim_ubd_new + 0.03
    else:
        xlim_ubd_new = xlim_ubd_new + 0.015
    ax.set_xlim(0, xlim_ubd_new)
    my_lbs = ax.get_xticks().tolist()
    my_lbs1 = [str(int(ele * 100)) for ele in my_lbs]
    ax.set_xticklabels(my_lbs1, fontsize=30)
    ylim_ubd = net_benefit_all.max()
    yax_cut = [ylim_ubd / 5 * i for i in range(6)]
    ax.set_yticks(yax_cut)
    ylbs = [item.get_text() for item in ax.get_yticklabels()]
    ylbs = ['0', '20', '40', '60', '80', '100']
    ax.set_title(target_label_dict[target_code], y=1.0, pad=-36, fontsize=42)
    ax.set_yticklabels(ylbs, fontsize=30)
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.legend().set_visible(False)
    plt.margins(0.0, 0.05)
    plt.tight_layout()
    plt.savefig(outpath + 'SuppData/sFigures/sFigure5/' + target_code + '.png', bbox_inches='tight', pad_inches=0.05)


# Remind to customize codes for breast and prostate cancer
