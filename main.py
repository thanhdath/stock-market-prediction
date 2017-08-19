import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import sys

# import pdb

def showGraph(data):
    plt.figure()
    plt.plot(data)
    plt.show()
def saveGraph(data, file_name):
    plt.figure(dpi=360)
    plt.plot(data)
    plt.savefig('results/' + file_name)
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
    return pandas.read_csv('datas/' + file_name, usecols=[1], sep='|')

def normal_neural(data_file, result_path):
    print('1. Normal Neural:')
    dataset = read_data(data_file)
    saveGraph(dataset, result_path + '/data')

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset.loc[-1] = [max(dataset.values)*2]
    dataset = scaler.fit_transform(dataset)
    dataset = dataset[:-1]

    train_size = int(len(dataset) * 0.9)
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

    print('----- Predict Long Time -----')
    # predict long time
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)
    trainYTranform = scaler.inverse_transform([trainY])
    testYTransform = scaler.inverse_transform([testY])
    # calculate relative error
    relativeErrorTrain = relative_error(trainYTranform[0], trainPredict[:,0])
    print('Train Relative Error: %.2f ' % (relativeErrorTrain))
    relativeErrorTest = relative_error(testYTransform[0], testPredict[:,0])
    print('Test Relative Error: %.2f ' % (relativeErrorTest))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.figure(dpi=360)
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.savefig('results/' + result_path + '/normal_neural')


    # loop predict only the next day and fit to model
    print('----- Predict Trend -----')
    predict = []

    # pdb.set_trace()

    for index, today_close_price in enumerate(testX):
        predict_tomorrow = model.predict(today_close_price)[0]
        predict.append(predict_tomorrow)
        model.fit(today_close_price, numpy.array([testY[index]]), epochs=100, batch_size=1, verbose=0)

    predict = scaler.inverse_transform(predict)
    testX = scaler.inverse_transform(testX)
    testY = scaler.inverse_transform([testY])

    predict = [x[0] for x in predict]
    testX = [x[0] for x in testX]

    with open('results/' + result_path + '/result_normal_neural.csv', 'w+') as file:
        file.write('today,tomorrow_reality,tomorrow_predict\n')
        for index in range(len(testX)):
            file.write(str(testX[index]) + '|' + str(testY[0][index])  + '|' + str(predict[index]))
            file.write('\n')

    trend_reallity = [numpy.sign(y - x) for x, y in zip(testX, testY[0])]
    trend_predict = [numpy.sign(y - x) for x, y in zip(testX, predict)]

    number_correct = sum([(0, 1)[x == y] for x, y in zip(trend_reallity, trend_predict)])
    percent_correct = number_correct * 100 / len(testX)
    print('Percent Correct: %.2f%%' % percent_correct)
    plt.figure(dpi=360)
    plt.plot(testY)
    plt.plot(predict)
    plt.savefig('results/' + result_path + '/normal_neural_trending')
    with open('results/' + result_path + '/result_normal_neural_percent.txt', 'w+') as file:
        file.write(str(percent_correct))

def normal_neural_not_scaler():
    print('1. Normal Neural Not Scaler:')
    # Just test
    # dataset = read_data('data_stock_market.csv')
    # saveGraph(dataset, 'not_scaler/data')

    # dataset = numpy.array(dataset)

    # train_size = int(len(dataset) * 0.9)
    # test_size = len(dataset) - train_size
    # train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # look_back = 1
    # trainX, trainY = create_dataset(train, look_back)
    # testX, testY = create_dataset(test, look_back)

    # model = Sequential()
    # model.add(Dense(1, input_dim=1))
    # model.add(Dense(4, activation='sigmoid'))
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)

    # print('----- Predict Long Time -----')
    # # predict long time
    # # make predictions
    # trainPredict = model.predict(trainX)
    # testPredict = model.predict(testX)
    # # calculate relative error
    # relativeErrorTrain = relative_error(trainY, trainPredict[:,0])
    # print('Train Relative Error: %.2f ' % (relativeErrorTrain))
    # relativeErrorTest = relative_error(testY, testPredict[:,0])
    # print('Test Relative Error: %.2f ' % (relativeErrorTest))
    # # shift train predictions for plotting
    # trainPredictPlot = numpy.empty_like(dataset)
    # trainPredictPlot[:, :] = numpy.nan
    # trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # # shift test predictions for plotting
    # testPredictPlot = numpy.empty_like(dataset)
    # testPredictPlot[:, :] = numpy.nan
    # testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # # plot baseline and predictions
    # plt.figure(dpi=360)
    # plt.plot(dataset)
    # plt.plot(trainPredictPlot)
    # plt.plot(testPredictPlot)
    # plt.savefig('results/normal_neural')


    # # loop predict only the next day and fit to model
    # print('----- Predict Trend -----')
    # predict = []

    # # pdb.set_trace()

    # for index, today_close_price in enumerate(testX):
    #     predict_tomorrow = model.predict(today_close_price)[0]
    #     predict.append(predict_tomorrow)
    #     model.fit(today_close_price, numpy.array([testY[index]]), epochs=100, batch_size=1, verbose=0)

    # testY = [testY]
    # predict = [x[0] for x in predict]
    # testX = [x[0] for x in testX]

    # with open('results/result_normal_neural.csv', 'w+') as file:
    #     file.write('today,tomorrow_reality,tomorrow_predict\n')
    #     for index in range(len(testX)):
    #         file.write(str(testX[index]) + '|' + str(testY[0][index])  + '|' + str(predict[index]))
    #         file.write('\n')

    # trend_reallity = [numpy.sign(y - x) for x, y in zip(testX, testY[0])]
    # trend_predict = [numpy.sign(y - x) for x, y in zip(testX, predict)]

    # number_correct = sum([(0, 1)[x == y] for x, y in zip(trend_reallity, trend_predict)])
    # percent_correct = number_correct * 100 / len(testX)
    # print('Percent Correct: %.2f%%' % percent_correct)
    # plt.figure(dpi=360)
    # plt.plot(testY)
    # plt.plot(predict)
    # plt.savefig('results/not_scaler/normal_neural_trending')
    # with open('results/not_scaler/result_normal_neural_percent.txt', 'w+') as file:
    #     file.write(str(percent_correct))

def lstm(datafile, result_path):
    print('2. LSTM ')
    dataset = read_data(datafile)
    saveGraph(dataset, result_path + '/data')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset.loc[-1] = [max(dataset.values)*2]
    dataset = scaler.fit_transform(dataset)
    dataset = dataset[:-1]

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    look_back = 5
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    # pdb.set_trace()
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(5, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    # make predictions

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredictTransform = scaler.inverse_transform(trainPredict)
    trainYTransform = scaler.inverse_transform([trainY])
    testPredictTransform = scaler.inverse_transform(testPredict)
    testYTransform = scaler.inverse_transform([testY])
    # calculate root mean squared error
    # trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    trainScore = relative_error(trainYTransform[0], trainPredictTransform[:,0])
    print('Train Score: %.2f ' % (trainScore))
    # testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    testScore = relative_error(testYTransform[0], testPredictTransform[:,0])
    print('Test Score: %.2f ' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredictTransform)+look_back, :] = trainPredictTransform
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredictTransform)+(look_back*2)+1:len(dataset)-1, :] = testPredictTransform
    # plot baseline and predictions
    plt.figure(dpi=360)
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.savefig('results/' + result_path + '/long_time')

    # loop predict only the next day and fit to model
    print('----- Predict Trend -----')
    predict = []

    for index, today_close_price in enumerate(testX):
        predict_tomorrow = model.predict(numpy.array([today_close_price]))[0]
        predict.append(predict_tomorrow)
        model.fit(numpy.array([today_close_price]), numpy.array([testY[index]]), epochs=100, batch_size=1, verbose=0)

    predict = scaler.inverse_transform(predict)
    testX = scaler.inverse_transform(testX[:, 0])
    testY = scaler.inverse_transform([testY])

    predict = [x[0] for x in predict]
    testX = [x[look_back-1] for x in testX]
    testY = testY[0]

    with open('results/' + result_path + '/result_lstm_trending.csv', 'w+') as file:
        file.write('today,tomorrow_reality,tomorrow_predict\n')
        for index in range(len(testX)):
            file.write(str(testX[index]) + '|' + str(testY[index])  + '|' + str(predict[index]))
            file.write('\n')

    trend_reallity = [numpy.sign(y - x) for x, y in zip(testX, testY)]
    trend_predict = [numpy.sign(y - x) for x, y in zip(testX, predict)]

    number_correct = sum([(0, 1)[x == y] for x, y in zip(trend_reallity, trend_predict)])
    percent_correct = number_correct * 100 / len(testX)
    print(model.get_weights())
    print('Percent Correct: %.2f%%' % percent_correct)
    plt.figure(dpi=360)
    plt.plot(testY)
    plt.plot(predict)
    plt.savefig('results/' + result_path + '/trending')
    with open('results/' + result_path + '/result_lstm.txt', 'w+') as file:
        file.write(str(percent_correct))

if __name__== '__main__':
    data_vnindex = 'data_stock_market.csv'
    data_sp500 = 'SP500_15082017.csv'
    data_nasdaq = 'Nasdaq_15082017.csv'
    data_downjone = 'DownJone_15082017.csv'

    if sys.argv[2] == 'vn-index':
        datafile, result_path = data_vnindex, 'lstm_vnindex'
    elif sys.argv[2] == 'sp500':
        datafile, result_path = data_sp500, 'lstm_sp500'
    elif sys.argv[2] == 'nasdaq':
        datafile, result_path = data_nasdaq, 'lstm_nasdaq'
    elif sys.argv[2] == 'downjone':
        datafile, result_path = data_downjone, 'lstm_downjone'

    if sys.argv[1] == 'normal-neural':
        normal_neural(datafile, 'feed_forward')
    elif sys.argv[1] == 'normal-neural-not-scaler':
        normal_neural_not_scaler()
    elif sys.argv[1] == 'lstm':
        lstm(datafile, result_path)
