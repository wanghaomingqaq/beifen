import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import numpy as np
import matplotlib
from keras.models import load_model
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
model = load_model('model19.h5')
history=model.fit(np.array(X_train),np.array(X_train),
                  batch_size=20,
                  epochs=2000,
                  validation_data=(np.array(X_verify),np.array(X_verify)),
                  verbose=1)

mp = "./model20.h5"


model.save(mp)