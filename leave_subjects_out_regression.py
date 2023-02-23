import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from window_average import wind_avg
import scipy.io as spio
import pandas as pd
'''
Run file to perform leave subject out analysis
XC: Changes in Bio-Z Features from first point
YC: Changes in BP from first point
XD: Raw Bio-Z Features
YD: Raw in BP
'''
matData = spio.loadmat('Feature_and_BP_Subject_Data.mat')
df = pd.DataFrame({'XC': matData['XC'][0], 'YC': matData['YC'][0], 'ZD': matData['ZD'][0], 'YD': matData['YD'][0]})

test_subjects = [9] # Select which subject you want in the testing set
# Z_ = Bio-Z Data, Y_ = BP Data, YC_ = Change in BP Data
Z_test, Y_test, YC_test = np.empty((0, 15)), np.empty((0, 3)), np.empty((0, 3))
for t in test_subjects:
    Z_test = np.append(Z_test, df['XC'].iloc[t - 1][0], axis=0)
    Y_test = np.append(Y_test, df['YD'].iloc[t - 1][0], axis=0)
    YC_test = np.append(YC_test, df['YC'].iloc[t - 1][0], axis=0)

train_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 10]  # Select which subjects you want in the training set
Z_train, YC_train = np.empty((0, 15)), np.empty((0, 3))
for t in train_subjects:
    Z_train = np.append(Z_train, df['XC'].iloc[t - 1][0], axis=0)
    YC_train = np.append(YC_train, df['YC'].iloc[t - 1][0], axis=0)

# For subject 9 33% of their data is automatically added in training
if test_subjects[0] == 9:
    # Employ for Subject 9
    l = int(len(Z_test) * 0.33)
    Z_train = np.append(Z_train, Z_test[:l, :], axis=0)
    YC_train = np.append(YC_train, YC_test[:l, :], axis=0)

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

# Plot training loss curve
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# Apply moving average filter to testing data
if test_subjects[0] == 9:
    Z_test, Y_test = wind_avg(10, 5, Z_test[l:, :], 15, Y_test[l:, :], 3)
else: 
    Z_test, Y_test = wind_avg(10, 5, Z_test, 15, Y_test, 3)
 
Ypredicted = ann.predict(Z_test) # Predict BP
plt.figure()
# 0 - SBP, 1 - DBP
for BP in (0, 2):
    cal = np.mean(Y_test[[-1, 1], BP] - Ypredicted[[-1, 1], BP])
    Ypredicted[:, BP] = Ypredicted[:, BP] + cal
    if BP == 0:
        print('SBP Error:', np.mean(Y_test[:, BP] - Ypredicted[:, BP]), np.std(Y_test[:, BP] - Ypredicted[:, BP]))
    else:
        print('DBP Error:', np.mean(Y_test[:, BP] - Ypredicted[:, BP]), np.std(Y_test[:, BP] - Ypredicted[:, BP]))
    plt.plot(Ypredicted[:, BP]), plt.plot(Y_test[:, BP])
