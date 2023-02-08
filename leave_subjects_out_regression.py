import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from window_average import wind_avg
import scipy.io as spio
import pandas as pd
'''
XC: Changes in Bio-Z Features from first point
YC: Changes in BP from first point
XD: Raw Bio-Z Features
YD: Raw in BP
'''
matData = spio.loadmat('DJ_PaperData.mat')
df = pd.DataFrame({'XC': matData['XC'][0], 'YC': matData['YC'][0], 'ZD': matData['ZD'][0], 'YD': matData['YD'][0]})

test_subjects = [7]
# Z_ = Bio-Z Data, Y_ = BP Data, YC_ = Change in BP Data
Z_test, Y_test, YC_test = np.empty((0, 15)), np.empty((0, 3)), np.empty((0, 3))
for t in test_subjects:
    Z_test = np.append(Z_test, df['XC'].iloc[t - 1][0], axis=0)
    Y_test = np.append(Y_test, df['YD'].iloc[t - 1][0], axis=0)
    YC_test = np.append(YC_test, df['YC'].iloc[t - 1][0], axis=0)

train_subjects = [1, 2, 3, 4, 5, 6, 8, 9, 10]
Z_train, YC_train = np.empty((0, 15)), np.empty((0, 3))
for t in train_subjects:
    Z_train = np.append(Z_train, df['XC'].iloc[t - 1][0], axis=0)
    YC_train = np.append(YC_train, df['YC'].iloc[t - 1][0], axis=0)

'''
# Employ for Subject 9
l = int(len(Z_data) * 0.33)
Z_data1 = np.append(Z_data1, Z_data[:l, :], axis=0)
Y_data1 = np.append(Y_data1, YC_data[:l, :], axis=0)
'''
#Build and train ANN model
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Normalization())
ann.add(tf.keras.layers.Input(shape=(15,)))
ann.add(tf.keras.layers.Dense(units=32, activation=tf.keras.layers.LeakyReLU(0.2)))
ann.add(tf.keras.layers.BatchNormalization())
ann.add(tf.keras.layers.Dense(units=8, activation=tf.keras.layers.LeakyReLU(0.2)))
ann.add(tf.keras.layers.Dense(units=3, activation='linear'))
from tensorflow.keras.optimizers import Adam
ann.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss=tf.keras.losses.MeanSquaredError())
history = ann.fit(Z_train, YC_train, batch_size=32, epochs=64)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
'''
# Employ for subject 9
Ztest, Ytest = wind_avg(10, 5, Z_data[l:, :], 15, Y_data[l:, :], 3)
'''
Z_test, Y_test = wind_avg(10, 5, Z_test, 15, Y_test, 3)
Ypredicted = ann.predict(Z_test)
plt.figure()
for BP in (0, 2):
    cal = np.mean(Y_test[[-1, 1], BP] - Ypredicted[[-1, 1], BP])
    Ypredicted[:, BP] = Ypredicted[:, BP] + cal
    if BP == 0:
        print('SBP Error:', np.mean(Y_test[:, BP] - Ypredicted[:, BP]), np.std(Y_test[:, BP] - Ypredicted[:, BP]))
    else:
        print('DBP Error:', np.mean(Y_test[:, BP] - Ypredicted[:, BP]), np.std(Y_test[:, BP] - Ypredicted[:, BP]))
    plt.plot(Ypredicted[:, BP]), plt.plot(Y_test[:, BP])
