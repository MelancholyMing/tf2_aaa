import numpy as np
import sys

h, l ,c = np.loadtxt('data.csv',delimiter=',',usecols=(4,5,6),unpack=True)
N = int(sys.argv[1])
h = h[-N:]
l = l[-N:]

print('len(h)',len(h),'len(l)',len(l))
print('close:',c)
previousclose = c[-N -1:-1]

print('len(previousclose):',len(previousclose))
print('previous colse:',previousclose)
# truerange = np.maximum(h-l,h-previousclose,previousclose-l)
truerange = np.maximum(np.maximum(h-l,h-previousclose),previousclose-l)
print('1:',h-l)
print('2:',h-previousclose,'\n')
print('3:',previousclose-l,'\n')
print('Truerange:', truerange,'\n')

atr = np.zeros(N)
atr[0] = np.mean(truerange)
for i in range(1,N):
    atr[i] = (N-1)*atr[i-1] + truerange[i]
    atr[i] /= N
print("ATR:",atr)
