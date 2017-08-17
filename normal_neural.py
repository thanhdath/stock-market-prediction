import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys

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
def read_data(file_name):
    return pandas.read_csv(file_name, usecols=[1])

if __name__== '__main__':
    dataset = read_data('data_stock_market.csv')
    # showGraph(dataset)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset.loc[-1] = [max(dataset.values)*2]
    dataset = scaler.fit_transform(dataset)
    dataset = dataset[:-1]

    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    model = Sequential()
    model.add(Dense(1, input_dim=1))
    model.add(Dense(4, activation='sigmoid'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)

    if sys.argv[-1] == '1':
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
    else:
        predict = []

        for index, today_close_price in enumerate(testX):
            predict_tomorrow = model.predict(today_close_price)[0]
            predict.append(predict_tomorrow)
            model.fit(today_close_price, numpy.array([testY[index]]), epochs=100, batch_size=1, verbose=0)

        predict = scaler.inverse_transform(predict)
        testX = scaler.inverse_transform(testX)
        testY = scaler.inverse_transform(testY)

        predict = [x[0] for x in predict]
        testX = [x[0] for x in testX]

        with open('results/result_normal_neural.csv', 'w+') as file:
            file.write('today,tomorrow_reality,tomorrow_predict\n')
            for index in range(len(testX)):
                file.write(str(testX[index]) + ',' + str(testY[index])  + ',' + str(predict[index]))
                file.write('\n')

        trend_reallity = [numpy.sign(y - x) for x, y in zip(testX, testY)]
        trend_predict = [numpy.sign(y - x) for x, y in zip(testX, predict)]

        number_correct = sum([(0, 1)[x == y] for x, y in zip(trend_reallity, trend_predict)])
        percent_correct = number_correct * 100 / len(testX)
        print('Percent Correct: %.2f%%' % percent_correct)
