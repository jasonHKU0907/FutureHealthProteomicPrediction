

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
from sklearn.metrics import recall_score, roc_auc_score, mean_squared_error, mean_absolute_error
import glob
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Results/MLModeling/MLP/SharedHead/'
fold_id_lst = [ele for ele in range(10)]

pro_df = pd.read_csv(dpath + 'PreprocessedData/ProteomicsData/ProteomicsData_imp.csv')
pro_f_lst = pro_df.columns.tolist()[1:]

target_df = pd.read_csv(dpath + 'PreprocessedData/TargetData/TargetData.csv')
target_code_lst = ['A0', 'C0', 'D0', 'E0', 'F0', 'G0', 'H0', 'H1', 'I0', 'J0', 'K0', 'L0', 'M0', 'N0']
target_f_lst = [target_code + '_y' for target_code in target_code_lst]
target_yrs_lst = [target_code + '_years' for target_code in target_code_lst]

eid_lst, region_lst = [], []
comb_bef_lst, comb_aft_lst = [], []

for i in range(len(target_df)):
    eid_lst.append(int(target_df.iloc[i].eid))
    region_lst.append(int(target_df.iloc[i].Region_code))
    ind_y_df = target_df[target_f_lst].iloc[i,:]
    ind_yrs_df = target_df[target_yrs_lst].iloc[i,:]
    ind_df = np.array(ind_y_df)*np.array(ind_yrs_df).tolist()
    comb_bef_lst.append(len(ind_df[ind_df < 0]))
    comb_aft_lst.append(len(ind_df[ind_df > 0]))

my_target = pd.DataFrame({'eid':eid_lst, 'Region_code':region_lst, 'combid_y_bef': comb_bef_lst, 'combid_y_aft': comb_aft_lst})
mydf = pd.merge(pro_df, my_target[['eid', 'Region_code', 'combid_y_bef', 'combid_y_aft']], how = 'inner', on = ['eid'])
mydf['combid_y_bef'] = mydf['combid_y_bef']/14
mydf['combid_y_aft'] = mydf['combid_y_aft']/14
X, y_bef, y_aft = mydf[pro_f_lst].copy(), mydf.combid_y_bef.copy(), mydf.combid_y_aft.copy()


def get_MLP():
    inputs = Input((1461, ), name="a_inputs")
    Dense1 = Dense(512, activation='relu', name="a_Dense1")(inputs)
    Dense2 = Dense(256, activation='relu', name="a_Dense2")(Dense1)
    Dense3 = Dense(128, activation='relu', name="a_Dense3")(Dense2)
    Dense4_1 = Dense(64, activation='relu', name="a_Dense4_1")(Dense3)
    Dense4_2 = Dense(64, activation='relu', name="a_Dense4_2")(Dense3)
    outputs1 = Dense(1, activation='linear', name="a_bef")(Dense4_1)
    outputs2 = Dense(1, activation='linear', name="a_aft")(Dense4_2)
    model = Model(inputs, [outputs1, outputs2])
    optimizer = tf.keras.optimizers.Adam(1e-5)
    losses = {'a_bef': 'mse', 'a_aft': 'mse'},
    loss_weights = {'a_bef': 1, 'a_aft': 1},
    metrics = {'a_bef': 'mse', 'a_aft': 'mse'}
    model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer, metrics=metrics)
    return model


for fold_id in fold_id_lst:
    model = get_MLP()
    mc_file = outpath + 'SH_fold' + str(fold_id) + '.h5'
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    X_train0, X_test = np.array(X.iloc[train_idx, :]), np.array(X.iloc[test_idx, :])
    y_bef_train0, y_bef_test = np.array(y_bef.iloc[train_idx]), np.array(y_bef.iloc[test_idx])
    y_aft_train0, y_aft_test = np.array(y_aft.iloc[train_idx]), np.array(y_aft.iloc[test_idx])
    ids, nb_train = np.arange(len(X_train0)), int(len(X_train0) * 0.8)
    np.random.shuffle(ids)
    X_train, y_bef_train, y_aft_train = X_train0[ids[:nb_train], :], y_bef_train0[ids[:nb_train]], y_aft_train0[ids[:nb_train]]
    X_val, y_bef_val, y_aft_val = X_train0[ids[nb_train:], :], y_aft_train0[ids[nb_train:]], y_aft_train0[ids[nb_train:]]
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    mc = ModelCheckpoint(filepath=mc_file, monitor='val_loss', save_best_only=True)
    model.fit(X_train, [y_bef_train, y_aft_train], validation_data=(X_val, [y_bef_val, y_aft_val]), epochs=1000, batch_size=128, verbose=1, callbacks=[es, mc])

