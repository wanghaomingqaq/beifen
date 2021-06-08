import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('./data/dataset03.csv',nrows=150)
import matplotlib
from matplotlib.pyplot import *
# data2 = pd.read_csv('./data/dataset04.csv',index_col='DATETIME')
# data2_X= data2['10/09/16 02':'19/09/16 10'] # 13/09/16 23   16/09/16 00
font = {'family': 'Microsoft YaHei',  'weight': 'bold',  'size': '14'}
matplotlib.rc("font", family="Microsoft YaHei", weight="bold", size="14")
print(data)
# plt.plot(data.S_PU8, color = 'black')
# plt.plot(data.S_PU11, color = 'blue')
ax = plt.gca()
ax.set_ylim(0,8)

plt.plot(data.S_V2*5, color='black',label='阀门V2',ls='-.')
# plt.plot(data.S_PU11,color='yellow')
plt.plot(data.L_T2,color='red',label='水箱2')
legend(loc=0,)
plt.ylabel("流量")
# plt.plot(data2.S_PU2, color = 'green')
# plt.plot(data2.L_T7,color = 'red')
# plt.plot(data2_X.F_PU9)
# plt.axvline('13/09/16 23',color='red')
# plt.axvline('16/09/16 00',color='red')
plt.show()
