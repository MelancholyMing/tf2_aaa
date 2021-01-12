import datetime
import numpy as np
import matplotlib.dates as mdates
import pandas_datareader.data as webdata
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf

yf.pdr_override()  # 因为雅虎在中国受限制的原因，需要引入另一个模块，“yfinance”

startdate = datetime.datetime(2014, 4, 12)
enddate = datetime.datetime(2015, 5, 12)
# today = enddate = datetime.date.today()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 允许中文
plt.rcParams['axes.unicode_minus'] = False  # 允许有坐标轴中的正负号
plt.rc('axes', grid=True)
plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)
rect = [0.4, 0.5, 0.8, 0.5]

fig = plt.figure(facecolor='white', figsize=(12, 11))

# axescolor = '#f6f6f6' #the axes background color
ax = fig.add_axes(rect, facecolor='#f6f6f6')  # left, bottom, width, height =[0.4,0.5,0.8,0.5]
ax.set_ylim(10, 800)


def plotTicker(ticker, startdate, enddate, fillcolor):
    r = webdata.get_data_yahoo(ticker, startdate, enddate)
    print(r.head())
    mpf.plot(r, type='candle', mav=(2, 5, 10), volume=True)
    ###plot the relative strength indicator
    ###adjusted close removes the impacts of splits and dividends
    prices = r['Adj Close']

    ###plot the price and volume data

    ax.plot(r.index, prices, color=fillcolor, lw=2, label=ticker)
    ax.legend(loc='upper right', shadow=True, fancybox=True)  # shadow: 是否为图例边框添加阴影,fancybox: 是否将图例框的边角设为圆形
    '''
    #set the labels rotation and alignment
    for label in ax.get_xtickabels():    #get_xtickabels()也是用不了了，所以我直接注释了
        #To display date label slanting at 30 degreees
        label.set_rotation(30)
        label.set_horizontalalignment('right')
    '''
    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')


# plot the tickers now
plotTicker('BIDU', startdate, enddate, 'red')
plotTicker('GOOG', startdate, enddate, '#1066ee')
plotTicker('AMZN', startdate, enddate, '#506612')

plt.show()
