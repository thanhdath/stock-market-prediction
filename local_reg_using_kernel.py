import numpy as np
import matplotlib.pyplot as plt
import pandas
import math

def showGraph(*args):
    for arg in args:
        plt.plot(arg)
    plt.show()

data_set = np.array(pandas.read_csv('data_stock_market.csv', usecols=[1])).reshape(-1)

total_length = data_set.shape[0]
train_length = int(total_length*0.8)
x_train = np.array(range(1,train_length + 1))
y_train = data_set[0: train_length]

x_test = np.array(range(train_length + 1, total_length + 1))
y_test = data_set[train_length:]

def LWLR_predict(test_point, x_train, y_train, k = 10.0):
    X_train = np.mat(x_train)
    Y_train = np.mat(y_train).T
    m = X_train.shape[1]
    W_ = np.mat(np.eye((m)))
    for i in range(m):
        diffMat = test_point - X_train[0,i]
        W_[i,i] = math.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = X_train.T*W_*X_train
    if np.linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    W = xTx.I * (X_train.T * (W_ * Y_train))
    return test_point * W

def predict(test_set, x_train, y_train, k = 10.0):
    m = test_set.shape[0]
    Y_pred = np.zeros(m)
    for i in range(m):
        Y_pred[i] = LWLR_predict(test_set[i], x_train, y_train, k)
    return Y_pred

y_pred = predict(x_test, x_train, y_train)
print(y_pred)
print(y_test)
showGraph(y_pred, y_test)
abs_error = np.absolute(np.subtract(y_pred - y_test))
relative_error = np.divide(abs_error, y_test)
print(relative_error)
mean_error = np.mean(relative_error)
print(mean_error)

