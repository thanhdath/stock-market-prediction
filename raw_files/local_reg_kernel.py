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
X = np.array(range(1, total_length + 1))
Y = data_set

def LWLR_predict(test_point, x_train, y_train, k = 10.0):
    X_train = x_train.reshape([-1,1])
    Y_train = y_train.reshape([-1,1])
    m = X_train.shape[0]
    W_ = np.eye((m))
    threshold = m
    for i in range(m)[::-1]:
        diff = test_point - X_train[i,0]
        W_[i,i] = math.exp(diff*diff/(-2.0*k**2))
        if W_[i,i] > 0.01:
            threshold = i
        else:
            break
    # print("threshold:", threshold)
    X_train = X_train[threshold - 1:,:]
    # print("X_train.shape:", X_train.shape)
    Y_train = Y_train[threshold - 1:,:]
    # print("Y_train.shape:", Y_train.shape)
    W_ = W_[threshold - 1:, threshold -1:]
    # print("W_.shape:", W_.shape)
    xTx = np.matmul(np.matmul(np.transpose(X_train), W_), X_train)
    # print("xTx.shape:", xTx.shape)
    if np.linalg.det(xTx) == 0.0:
        # print ("This matrix is singular, cannot do inverse")
        return
    W = np.linalg.inv(xTx)*np.matmul(np.matmul(np.transpose(X_train), W_), Y_train)
    print(W) 
    return test_point * W

def predict(data_set, i, k = 10.0):
    if i == 0:
        return data_set[0], 0.0
    
    train_y = data_set[:i]
    train_x = np.array(range(1, i+1))
    y_pred = LWLR_predict(i+1, train_x, train_y, k)
    y_true = data_set[i]
    abs_error = np.absolute(y_pred - y_true)
    rel_error = (float)(abs_error)/y_true
    print(i)
    print("true_value:", y_true)
    print("pred_value:", y_pred)
    print("relative_error:", rel_error)
    print("=================================")
    return y_pred, rel_error

pred_values = []
rel_errors = []
for i in range(data_set.shape[0]):
    pred_value, rel_error = predict(data_set, i)
    pred_values.append(pred_value)
    rel_errors.append(rel_error)

print ("relative_error:", np.mean(np.array(rel_errors)))

plt.plot(data_set)
plt.plot(pred_values)
plt.show()
