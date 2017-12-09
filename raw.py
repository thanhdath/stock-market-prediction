import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas
import numpy as np
# from sklearn import linear_model
import datetime as dt


            
dataset = pandas.read_csv('data_new/DowJones2.csv')[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
x = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in dataset['Date']]

plt.figure(dpi=200)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=24))
# locator = mdates.MonthLocator(interval=6)
# locator.MAXTICKS = 10
plt.plot(x, dataset['Volume'])
plt.gcf().autofmt_xdate()
plt.title('Volume')
plt.show()

# # dataset = pandas.read_csv('data_new/VNIndex.csv')['Close']

# # dataset = dataset[::-1]
# # dataset.index = dataset.index.values[::-1]

# plt.figure(dpi=200)
# plt.plot(dataset, linewidth=.5, color='blue')
# for i in range(0, len(dataset), 10):
#     x = list(range(dataset.index.values[i], dataset.index.values[i]+10))
#     y = dataset[i:i+10]
#     print(x, y)
#     try:
#         fit = np.polyfit(x, y, 1)
#         fit_fn = np.poly1d(fit) 
#         # plt.plot(dataset[-100:], linewidth=.5)
#         # plt.plot(fit_fn, linewidth=.5)
#         plt.plot(x, fit_fn(x), color='orange', linewidth=1)
#     except Exception as e:
#         pass
# for i in range(0, len(dataset), 30):
#     x = list(range(dataset.index.values[i], dataset.index.values[i]+30))
#     y = dataset[i:i+30]
#     print(x, y)
#     try:
#         fit = np.polyfit(x, y, 1)
#         fit_fn = np.poly1d(fit) 
#         # plt.plot(dataset[-100:], linewidth=.5)
#         # plt.plot(fit_fn, linewidth=.5)
#         plt.plot(x, fit_fn(x), color='red', linewidth=1)
#     except Exception as e:
#         pass
# plt.show()
# plt.figure(dpi=100)
# x = list(range(5))
# y = dataset[:5]

# fit = np.polyfit(x, y, 1)
# print(fit)
# fit_fn = np.poly1d(fit) 
# # plt.plot(dataset[-100:], linewidth=.5)
# # plt.plot(fit_fn, linewidth=.5)
# plt.plot(y, color='blue')
# plt.plot(x, fit_fn(x), color='orange')

# x = list(range(5, 10))
# y = dataset[5:10]

# fit = np.polyfit(x, y, 1)
# print(fit)
# fit_fn = np.poly1d(fit) 
# # plt.plot(dataset[-100:], linewidth=.5)
# # plt.plot(fit_fn, linewidth=.5)
# plt.plot(y, color='blue')
# plt.plot(x, fit_fn(x), color='orange')

# x = list(range(10, 15))
# y = dataset[10:15]

# fit = np.polyfit(x, y, 1)
# print(fit)
# fit_fn = np.poly1d(fit) 
# # plt.plot(dataset[-100:], linewidth=.5)
# # plt.plot(fit_fn, linewidth=.5)
# plt.plot(y, color='blue')
# plt.plot(x, fit_fn(x), color='orange')

# plt.show()

# def standardlize(key, mul):
#     for i in range(len(dataset[key])):
#         dataset[key][i] *= 100*mul
#         dataset[key][i] = int(dataset[key][i])/mul



# dataset = pandas.read_csv('array.txt', sep='&')
# print(dataset.columns)

# dataset = dataset[['Unnamed: 0', ' Train APG ', ' Train PAD ',' Test APG ', ' Test PAD']]

# standardlize(' Train APG ', 1)
# standardlize(' Test APG ', 1)
# standardlize(' Train PAD ', 100)
# standardlize(' Test PAD', 100)



# for i in range(len(dataset.values)):
#     print(i+1,end='')
#     for j in range(1, len(dataset.values[i])):
#         print('&', dataset.values[i][j],end='')
#     print('\\\\\\hline')

keys = [ ' Train APG ', ' Train PAD ',' Test APG ', ' Test PAD']
dataset = pandas.read_csv('array.txt', sep='&')
print(dataset.columns)

dataset = dataset[keys]

print('Average', end='')
for key in keys:
    print('&', np.mean(dataset[key].values), end='')
print('\\\\\\hline')