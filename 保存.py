import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import numpy as np
from keras.models import load_model
import matplotlib
from matplotlib.pyplot import *
import seaborn as sns
# sns.set(color_codes=True)
font = {'family': 'Microsoft YaHei',  'weight': 'bold',  'size': '16'}
matplotlib.rc("font", family="Microsoft YaHei", weight="bold", size="16")

df = pd.read_csv('./data/dataset03.csv')
X_train = df.iloc[:,1:]
df_verify = pd.read_csv('./data/dataset04.csv')
X_verify = df.iloc[:,1:]
df_test = pd.read_csv('./data/test_dataset.csv')
X_test = df_test.iloc[:,1:]

model=Sequential()
model.add(Dense(64,activation='relu',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(0.0),
                input_shape=(X_train.shape[1],)
               )
         )
model.add(Dense(8,activation='relu',
                kernel_initializer='glorot_uniform'))
model.add(Dense(64,activation='relu',
                kernel_initializer='glorot_uniform'))
model.add(Dense(X_train.shape[1],
                kernel_initializer='glorot_uniform'))
model.compile(loss='mse',optimizer='adam')
print(model.summary())
history=model.fit(np.array(X_train),np.array(X_train),
                  batch_size=10,
                  epochs=300,
                  validation_data=(np.array(X_verify),np.array(X_verify)),
                  verbose=1)

mp = "./model.h5"


model.save(mp)
plt.gcf().subplots_adjust(bottom=0.15) # 字体显示完全
plt.plot(
    history.history['loss'],
    'b',
    color='red'

)

plt.legend(loc='upper right')
plt.xlabel('轮次')
plt.ylabel('损失函数')
plt.show()

X_pred = model.predict(np.array(X_train))
X_pred = pd.DataFrame(X_pred,
                      columns=X_train.columns)
print(X_pred)
X_pred.index = X_train.index
scored = pd.DataFrame(index=X_train.index)
scored['损失函数'] = np.mean(np.abs(X_pred-X_train), axis = 1)
plt.figure()
sns.distplot(scored['损失函数'],
             bins = 10,
             kde= True,
            color = 'black')
plt.xlim([0.0,1.5])
plt.xlabel('损失函数值',fontsize=10)
plt.ylabel('分布情况')
plt.show()


X_pred = model.predict(np.array(X_test))
X_pred = pd.DataFrame(X_pred,
                      columns=X_test.columns)
X_pred.index = X_test.index
threshod = 0.75
scored = pd.DataFrame(index=X_test.index)
scored['损失数值'] = np.mean(np.abs(X_pred-X_test), axis = 1)
scored['阈值'] = threshod
scored['Anomaly'] = scored['损失数值'] > scored['阈值']
print('yyyyyyyyyyyyyyyy')
scored.head()
print(scored.head())
scored.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['black','red'])
plt.xlabel('序号',fontsize=10)
plt.ylabel('损失数值')
plt.show()
print(type(scored['损失数值']))
guiyihua = []
for i in scored['Anomaly']:
    if i == True:
        guiyihua.append(2)
    else:
        guiyihua.append(0)
print(guiyihua)
ax = plt.gca()
ax.set_ylim(0,3)
plt.gcf().subplots_adjust(bottom=0.15)
plt.plot(guiyihua,color='gray',label='检测值')
plt.plot(X_test.ATT_FLAG,color='black',label='实际值')
plt.title('攻击识别效果图')
plt.xlabel('序号',fontsize=10)
legend(loc=0,)
plt.show()