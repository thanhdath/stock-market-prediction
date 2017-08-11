
# coding: utf-8

# In[ ]:


import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[ ]:


def showGraph(data):
    plt.plot(data)
    plt.show()
# convert dataset to x and y
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
def relative_error(xs, ys):
    error = 0
    zero = 0
    for i in range(len(xs)):
        if ys[i] != 0:
            error += abs(xs[i]-ys[i])*100 / ys[i]
        else:
            zero += 1
    error /= (len(xs) - zero)
    return error


# In[ ]:


dataset = pandas.read_csv('data_stock_market.csv', usecols=[1])
data_standard = dataset


# In[ ]:


showGraph(dataset)


# In[ ]:


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset.loc[-1] = [max(dataset.values)*2]
dataset = scaler.fit_transform(dataset)
dataset = dataset[:-1]
# showGraph(dataset)


# In[ ]:


# split into train set and test set
train_size = int(len(dataset) * 0.67)
# train_size = int(lent(dataset))
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]


# In[ ]:


# showGraph(train)
# showGraph(test)


# In[ ]:


look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[ ]:


# This algorithm don't use LSTM network
model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[ ]:


model.fit(trainX, trainY, epochs=100, batch_size=1)


# In[ ]:


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
trainScore = relative_error(trainY[0], trainPredict[:,0])
print('Train Score: %.2f ' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
testScore = relative_error(testY[0], testPredict[:,0])
print('Test Score: %.2f ' % (testScore))


# In[ ]:


# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[ ]:


# nexts = model.predict(testX[-1])
# tomorrow_close_price = nexts[0][0] * max(data_standard.values)
# print(tomorrow_close_price[0])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




