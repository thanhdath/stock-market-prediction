import numpy 
import matplotlib.pyplot as plt 
import pandas
import math
import tensorflow as tf 
import keras.optimizers

dataset = pandas.read_csv('data_stock_market.csv', usecols = [1])
X = np.array(dataset).reshape(-1)
total_length = X.shape[0]
X_train = X

