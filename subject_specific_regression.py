import numpy as np
import matplotlib.pyplot as plt
from window_average import wind_avg
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import scipy.io as spio
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

'''
XC: Changes in Bio-Z Features from first point
YC: Changes in BP from first point
XD: Raw Bio-Z Features
YD: Raw in BP
'''
matData = spio.loadmat('Feature_and_BP_Subject_Data.mat')
df = pd.DataFrame({'XC': matData['XC'][0], 'YC': matData['YC'][0], 'ZD': matData['ZD'][0], 'YD': matData['YD'][0]})
for s in range(1, 11):
    print('\n')
    print('Subject', s)
    subject = [s]
    # Z_data = Bio-Z Data, Y_data = BP Data
    Z_data, Y_data = np.empty((0, 15)), np.empty((0, 3))
    for t in subject:
        Z_data = np.append(Z_data, df['ZD'].iloc[t - 1][0], axis=0)
        Y_data = np.append(Y_data, df['YD'].iloc[t - 1][0], axis=0)
    Z_data, Y_data = wind_avg(10, 5, Z_data, 15, Y_data, 3)
    print('Range:', np.max(Y_data, axis=0), np.min(Y_data, axis=0))
    print('Mean/Std:', np.mean(Y_data, axis=0), np.std(Y_data, axis=0))
    print('N:', len(Y_data))

    # BP_Type 0 for SBP, 1 for MAP, and 2 for DBP
    for BP_Type in (0, 2):
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        ZBP, Y_data1 = np.array([]), np.array([])
        adaZ = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=5), n_estimators=50)
        for train_index, test_index in kf.split(Z_data):
            Z_train, Z_test = Z_data[train_index], Z_data[test_index]
            y_train, y_test = Y_data[train_index], Y_data[test_index]
            adaZ.fit(Z_train, y_train[:, BP_Type])
            Y_predicted = np.append(ZBP, adaZ.predict(Z_test))
            Y_real = np.append(Y_data1, y_test[:, BP_Type])

        plt.figure()
        sns.set_theme(), sns.set_context("talk"), sns.set_style('white')
        plt.plot(np.arange(1, len(Y_real) + 1), Y_real, '0.8', linewidth=6, marker='8')
        plt.plot(np.arange(1, len(Y_predicted) + 1), Y_predicted, linewidth=6)
        plt.legend(['Actual SBP', 'Predicted SBP', 'Actual DBP', 'Predicted DBP'], fontsize=14)
        plt.xlabel('Time [min]', fontsize=20), plt.ylabel('Blood Pressure [mmHg]', fontsize=20)
        plt.show()

        if BP_Type == 0:
            print('SBP Error: ', round(np.mean(Y_real - Y_predicted), 3), round(np.std(Y_real - Y_predicted), 3),
                  round(pearsonr(Y_real, Y_predicted)[0], 3), round(np.sqrt(np.mean(np.square(Y_real - Y_predicted))), 3))
        else:
            print('DBP Error: ', round(np.mean(Y_real - Y_predicted), 3), round(np.std(Y_real - Y_predicted), 3),
                  round(pearsonr(Y_real, Y_predicted)[0], 3), round(np.sqrt(np.mean(np.square(Y_real - Y_predicted))), 3))
