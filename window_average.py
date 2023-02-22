import numpy as np
def wind_avg(window_size, stride, X, X_feat_num, Y, Y_feat_num):
    ''' Function to apply a moving average to arrays'''
    X1, Y1 = np.array([]), np.array([])
    for f in range(0,X_feat_num):
        X2 = [np.mean(X[:,f][i:i+window_size]) for i in range(0, len(X[:,f]), stride) if i+window_size <= len(X[:,f]) ]
        if f == 0:
            X1 = np.append(X1, [X2])
        else:
            X1 = np.vstack([X1, X2])
    for g in range(0,Y_feat_num):
        Y2 = [np.mean(Y[:, g][i:i + window_size]) for i in range(0, len(Y[:, g]), stride) if i + window_size <= len(Y[:, g])]
        if g == 0:
            Y1 = np.append(Y1, [Y2])
        else:
            Y1 = np.vstack([Y1, Y2])
    Y = Y1.T
    X = X1.T
    return X, Y
