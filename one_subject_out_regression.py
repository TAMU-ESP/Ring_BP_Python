import scipy.io as spio
import numpy as np
import pandas as pd
from window_average import wind_avg
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

matData = spio.loadmat('DJ_PaperData.mat')
df = pd.DataFrame({'XC': matData['XC'][0], 'YC': matData['YC'][0], 'ZD': matData['ZD'][0], 'YD': matData['YD'][0]})

test_subject = [8]
# Z_ = Bio-Z Data, Y_ = BP Data
Z_test, Y_test = np.empty((0, 15)), np.empty((0, 3))
for t in test_subject:
    Z_test = np.append(Z_test, df['ZD'].iloc[t - 1][0], axis=0)
    Y_test = np.append(Y_test, df['YD'].iloc[t - 1][0], axis=0)
Z_test, Y_test = wind_avg(10, 5, Z_test, 15, Y_test, 3)

train_subjects = [6, 7, 9]
# Z_ = Bio-Z Data, Y_ = BP Data
Z_train, Y_train = np.empty((0, 15)), np.empty((0, 3))
for t in train_subjects:
    Z_train = np.append(Z_train, df['ZD'].iloc[t - 1][0], axis=0)
    Y_train = np.append(Y_train, df['YD'].iloc[t - 1][0], axis=0)
Z_train, Y_train = wind_avg(1, 1, Z_train, 15, Y_train, 3)
print('\nTest: Subject 8, Train: Subjects 6, 7, 9. Calibration: Last Point')

# BP_Type 0 for SBP, 1 for MAP, and 2 for DBP
for BP_Type in (0, 2):
    adaZ = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=4), n_estimators=20)
    adaZ.fit(Z_train, Y_train[:, BP_Type])
    Y_predicted = adaZ.predict(Z_test)
    cal = np.mean(Y_test[-1, BP_Type] - Y_predicted[-1])
    Y_predicted = Y_predicted + cal
    if BP_Type == 0:
        print('SBP Errors: ', np.mean(Y_test[:, BP_Type] - Y_predicted), np.std(Y_test[:, BP_Type] - Y_predicted),
              pearsonr(Y_test[:, BP_Type], Y_predicted)[0], np.sqrt(np.mean(np.square(Y_test[:, BP_Type] - Y_predicted))))
    else:
        print('DBP Errors: ', np.mean(Y_test[:, BP_Type] - Y_predicted), np.std(Y_test[:, BP_Type] - Y_predicted),
              pearsonr(Y_test[:, BP_Type], Y_predicted)[0], np.sqrt(np.mean(np.square(Y_test[:, BP_Type] - Y_predicted))))
    plt.figure()
    plt.plot(Y_test[:, BP_Type]), plt.plot(Y_predicted)