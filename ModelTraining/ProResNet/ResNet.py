

import os
import tensorflow as tf
import numpy as np
import random
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint,  EarlyStopping
from keras.models import load_model
from keras import backend as K
from Utility.Evaluation_Utilities import *
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix
pd.options.mode.chained_assignment = None  # default='warn'

def preprocess_target_individuals(target_df_file, target_code):
    target_y0, target_yrs0, target_source0 = target_code[0] + '0_y', target_code[0] + '0_years', target_code[0] + '0_source'
    target_y, target_yrs, target_source = target_code + '_y', target_code + '_years', target_code + '_source'
    target_df = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y, target_yrs])
    if (('C' in target_code) | ('X' in target_code)):
        target_df0 = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y0, target_yrs0])
        rm_idx0 = target_df0.index[target_df0[target_yrs0] <= 0]
        rm_eid_lst = target_df0.iloc[rm_idx0].eid.tolist()
    else:
        target_df0 = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y0, target_yrs0, target_source0])
        rm_idx0 = target_df0.index[(target_df0[target_source0] >= 50) | (target_df0[target_yrs0] <= 0)]
        rm_eid_lst = target_df0.iloc[rm_idx0].eid.tolist()
    clean_df = target_df[~target_df.eid.isin(rm_eid_lst)]
    clean_df.reset_index(inplace=True, drop = True)
    return clean_df


dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/MLModeling/MLP/'
target_df_file = dpath + 'PreprocessedData/TargetData/TargetData.csv'
cov_df = pd.read_csv(dpath + 'PreprocessedData/ProteomicsData/ProteomicsData_imp.csv')
cov_f_lst = cov_df.columns.tolist()[1:]
target_code_df = pd.read_csv(dpath + 'Target_code.csv', encoding='latin-1')
target_code_lst = target_code_df.Disease_code.tolist()
fold_id_lst = [ele for ele in range(10)]


def get_res_model():
    a_inputs = Input((1461,), name="a_inputs")
    a_Dense1 = Dense(512, activation='relu', name="a_Dense1")(a_inputs)
    a_Dense2 = Dense(256, activation='relu', name="a_Dense2")(a_Dense1)
    a_Dense3 = Dense(128, activation='relu', name="a_Dense3")(a_Dense2)
    a_Dense4_1 = Dense(64, activation='relu', name="a_Dense4_1")(a_Dense3)
    a_Dense4_2 = Dense(64, activation='relu', name="a_Dense4_2")(a_Dense3)
    b_inputs = Input((1461,), name="b_inputs")
    b_Dense1 = Dense(512, activation='relu', name="b_Dense1")(b_inputs)
    b_Dense2 = Dense(256, activation='relu', name="b_Dense2")(b_Dense1)
    b_Dense3 = Dense(128, activation='relu', name="b_Dense3")(b_Dense2)
    c_concat = concatenate([a_Dense4_1, a_Dense4_2, b_Dense3], name="c_concat")
    c_Dense1 = Dense(128, activation='relu', name="c_Dense1")(c_concat)
    c_Dense2 = Dense(64, activation='relu', name="c_Dense2")(c_Dense1)
    outputs = Dense(1, activation='sigmoid', name="outputs")(c_Dense2)
    model = Model([a_inputs, b_inputs], outputs)
    optimizer = tf.keras.optimizers.Adam(1e-5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

seed=2018

out_auc_lst = []

for target_code in target_code_lst:
    target_path = outpath + 'ResMLP/' + target_code + '/'
    os.mkdir(target_path)
    target_y, target_yrs = target_code + '_y', target_code + '_years'
    target_df = preprocess_target_individuals(target_df_file, target_code)
    mydf = pd.merge(cov_df, target_df, how = 'inner', on = ['eid'])
    X, y, y_yrs = mydf[cov_f_lst].copy(), mydf[target_y].copy(), mydf[target_yrs].copy()
    result_auc_cv, test_prs_df = [], pd.DataFrame()
    for fold_id in fold_id_lst:
        mc_file = target_path + 'MLP_fold' + str(fold_id) + '.h5'
        train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
        test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
        X_train0, X_test = np.array(X.iloc[train_idx, :]), np.array(X.iloc[test_idx, :])
        y_train0, y_test = np.array(y.iloc[train_idx]), np.array(y.iloc[test_idx])
        y_yrs_train0, y_yrs_test = np.array(y_yrs.iloc[train_idx]), np.array(y_yrs.iloc[test_idx])
        ids, nb_train = np.arange(len(X_train0)), int(len(X_train0)*0.8)
        np.random.shuffle(ids)
        X_train, y_train = X_train0[ids[:nb_train],:], y_train0[ids[:nb_train]]
        X_val, y_val = X_train0[ids[nb_train:],:], y_train0[ids[nb_train:]]
        res_model = get_res_model()
        sh_model = load_model(outpath + 'SharedHead/SH_fold' + str(fold_id) + '.h5')
        res_model.layers[1].set_weights(sh_model.layers[1].get_weights())
        res_model.layers[1].trainable = False
        res_model.layers[3].set_weights(sh_model.layers[2].get_weights())
        res_model.layers[3].trainable = False
        res_model.layers[5].set_weights(sh_model.layers[3].get_weights())
        res_model.layers[5].trainable = False
        res_model.layers[7].set_weights(sh_model.layers[4].get_weights())
        res_model.layers[7].trainable = False
        res_model.layers[8].set_weights(sh_model.layers[5].get_weights())
        res_model.layers[8].trainable = False
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
        mc = ModelCheckpoint(filepath=mc_file, monitor='val_loss', save_best_only=True)
        res_model.fit([X_train, X_train], y_train, validation_data=([X_val, X_val], y_val), epochs=1000, batch_size=128, verbose=1, callbacks=[es, mc])
        best_model = load_model(mc_file)
        y_pred_train = best_model.predict([X_train0, X_train0])[:, 0]
        train_prs_df = pd.DataFrame({'eid': mydf.iloc[train_idx].eid, 'Region_code': mydf.iloc[train_idx].Region_code, 'ProRS': y_pred_train, 'Y': y_train0, 'Y_yrs': y_yrs_train0})
        train_prs_df.to_csv(target_path + 'Train' + str(fold_id) + '_ProRS.csv', index=False)
        y_pred_test = best_model.predict([X_test, X_test])[:, 0]
        test_prs_df = pd.DataFrame({'eid':mydf.iloc[test_idx].eid, 'Region_code':[fold_id]*len(y_test), 'ProRS':y_pred_test, 'Y':y_test, 'Y_yrs':y_yrs_test})
        test_prs_df.to_csv(target_path + 'Test' + str(fold_id) + '_ProRS.csv', index=False)
        result_auc_cv.append(roc_auc_score(y_test, y_pred_test))
    out_auc_lst.append([target_code, np.mean(result_auc_cv), np.std(result_auc_cv)] + result_auc_cv)
    out_df = pd.DataFrame(out_auc_lst)
    out_df.columns = ['Disease_code', 'AUC_mean', 'AUC_std'] + ['auc_region' + str(ele) for ele in fold_id_lst]
    out_df = pd.merge(target_code_df, out_df, how='left', on=['Disease_code'])
    out_df.to_csv(outpath + 'ResMLP/Test_AUCs.csv', index=False)
    print((target_code, np.mean(result_auc_cv)))




'''
for F1
def preprocess_target_individuals(target_df_file, target_code):
    target_y0, target_yrs0, target_source0 = target_code[0] + '0_y', target_code[0] + '0_years', target_code[0] + '0_source'
    target_y, target_yrs, target_source = target_code + '_y', target_code + '_years', target_code + '_source'
    target_df = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y, target_yrs])
    if (('C' in target_code) | ('X' in target_code)):
        target_df0 = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y0, target_yrs0])
        rm_idx0 = target_df0.index[target_df0[target_yrs0] <= 0]
        rm_eid_lst = target_df0.iloc[rm_idx0].eid.tolist()
    else:
        target_df0 = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y0, target_yrs0, target_source0])
        rm_idx0 = target_df0.index[(target_df0[target_source0] >= 50) | (target_df0[target_yrs0] <= 0)]
        rm_eid_lst = target_df0.iloc[rm_idx0].eid.tolist()
    clean_df = target_df[~target_df.eid.isin(rm_eid_lst)]
    rm_idx = clean_df.index[clean_df[target_yrs] <= 0]
    clean_df.drop(rm_idx, axis = 0, inplace = True)
    clean_df.reset_index(inplace=True, drop = True)
    return clean_df
'''


'''
# C2/C3
def preprocess_target_individuals(target_df_file, target_code):
    target_y0, target_yrs0, target_source0 = target_code[0] + '0_y', target_code[0] + '0_years', target_code[0] + '0_source'
    target_y, target_yrs, target_source = target_code + '_y', target_code + '_years', target_code + '_source'
    target_df = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y, target_yrs])
    if (('C' in target_code) | ('X' in target_code)):
        target_df0 = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y0, target_yrs0])
        rm_idx0 = target_df0.index[target_df0[target_yrs0] <= 0]
        rm_eid_lst = target_df0.iloc[rm_idx0].eid.tolist()
    else:
        target_df0 = pd.read_csv(target_df_file, usecols=['eid', 'Region_code', target_y0, target_yrs0, target_source0])
        rm_idx0 = target_df0.index[(target_df0[target_source0] >= 50) | (target_df0[target_yrs0] <= 0)]
        rm_eid_lst = target_df0.iloc[rm_idx0].eid.tolist()
    clean_df = target_df[~target_df.eid.isin(rm_eid_lst)]
    clean_df.reset_index(inplace=True, drop = True)
    clean_df = clean_df.loc[clean_df.Gender != 1]
    #clean_df = clean_df.loc[clean_df.Gender == 1]
    clean_df.reset_index(inplace=True, drop = True)
    return clean_df
'''
