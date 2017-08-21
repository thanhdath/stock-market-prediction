
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math

from sklearn.preprocessing import MinMaxScaler

data_set = np.array(pandas.read_csv('data_stock_market.csv', usecols=[1])).reshape(-1)
total_length = data_set.shape[0]
X = np.array(range(1, total_length + 1))
Y = data_set

def predict(data_set, i):
    if i >= 10:
        train_y = data_set[i-10:i].reshape([-1,1])
        train_x = np.array(range(i-9,i+1)).reshape([-1,1])
    elif 0 < i < 10:
        train_y = data_set[0:i].reshape([-1,1])
        train_x = np.array(range(1,i+1)).reshape([-1,1])
    elif i <= 0:
        return data_set[i], 0

    xTx = np.linalg.inv(np.matmul(np.transpose(train_x),train_x))
    w = xTx * np.matmul(np.transpose(train_x),train_y)
    pred_y = w * (train_x[-1] + 1)
    abs_error = np.absolute(pred_y - data_set[i])
    rel_error = np.divide(abs_error, data_set[i])
    print(i)
    print("true_value:", data_set[i])
    print("pred_value:", pred_y.reshape(()))
    print("relative_error:", rel_error.reshape(()))
    print("=================================")
    return pred_y.reshape(()), rel_error.reshape(())

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





