import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import numpy as np
import seaborn as sns
sns.set(color_codes=True)


X_train = pd.read_csv('../data/dataset03.csv',usecols=['L_T1','F_PU1','S_PU1','S_PU2','F_PU2','S_PU3','F_PU3'])
X_test = pd.read_csv('../data/test_dataset.csv',usecols=['L_T1','F_PU1','S_PU1','S_PU2','F_PU2','S_PU3','F_PU3'])
X_verify = pd.read_csv('../data/dataset04.csv',usecols=['L_T1','F_PU1','S_PU1','S_PU2','F_PU2','S_PU3','F_PU3'])


model=Sequential()
model.add(Dense(16,activation='relu',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(0.0),
                input_shape=(X_train.shape[1],)
               )
         )
model.add(Dense(4,activation='relu',
                kernel_initializer='glorot_uniform'))
model.add(Dense(16,activation='relu',
                kernel_initializer='glorot_uniform'))
model.add(Dense(X_train.shape[1],
                kernel_initializer='glorot_uniform'))
model.compile(loss='mse',optimizer='adam')
print(model.summary())
history=model.fit(np.array(X_train),np.array(X_train),
                  batch_size=10,
                  epochs=100,
                  validation_data=(np.array(X_verify),np.array(X_verify)),
                  verbose=1)

plt.plot(
    history.history['loss'],
    'b',
    label='Training loss'
)
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss, [mse]')
plt.show()

X_pred = model.predict(np.array(X_train))
X_pred = pd.DataFrame(X_pred,
                      columns=X_train.columns)
print(X_pred)
X_pred.index = X_train.index
scored = pd.DataFrame(index=X_train.index)
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)
plt.figure()
sns.distplot(scored['Loss_mae'],
             bins = 10,
             kde= True,
            color = 'blue')
plt.xlim([0.0,.5])
plt.show()


X_pred = model.predict(np.array(X_test))
X_pred = pd.DataFrame(X_pred,
                      columns=X_test.columns)
X_pred.index = X_test.index
threshod = 0.4
scored = pd.DataFrame(index=X_test.index)
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis = 1)
scored['Threshold'] = threshod
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
scored.head()
scored.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','green'])
plt.show()
