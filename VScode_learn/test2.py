import pandas_datareader.data as webdata
from datetime import date
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.dates import DateFormatter
from matplotlib.dates import DayLocator
from matplotlib.dates import MonthLocator
from dateutil.relativedelta import relativedelta
today = date.today()
start = today- relativedelta(years=1)

quotes = webdata.get_data_yahoo("QQQ", start, today)
quotes = np.array(quotes)
dates = quotes.T[0]
qqq = quotes.T[4]
# 去除信号中的线性趋势。
y = signal.detrend(qqq)

alldays = DayLocator()
months = MonthLocator()
month_formatter = DateFormatter("%b %Y")

fig = plt.figure()
fig.subplots_adjust(hspace=.3)
ax = fig.add_subplot(211)

ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(month_formatter)

# 调大字号
ax.tick_params(axis='both', which='major', labelsize='x-large')

# 应用傅里叶变换，得到信号的频谱
amps = np.abs(fftpack.fftshift(fftpack.rfft(y)))

# 滤除噪声。如果某一频率分量的大小低于最强分量的10%，则将其滤除
amps[amps < 0.1*amps.max()] = 0

# 将滤波后的信号变换回时域，并和去除趋势后的信号一起绘制出来。
plt.plot(dates, y,'o', label='detrended')
plt.plot(dates, -fftpack.irfft(fftpack.ifftshift(amps)),label="filtered")

# 将x轴上的标签格式化为日期，并添加一个特大号的图例。
fig.autofmt_xdate()
plt.legend(prop={'size':'x-large'})

# 添加第二个子图，绘制滤波后的频谱。
ax2 = fig.add_subplot(212)
N = len(qqq)
plt.plot(np.linspace(-N/2, N/2, N), amps, label="transformed")

# 显示图像和图例
plt.legend(prop={'size':'x-large'})
plt.show()