import pandas as pd
import seaborn as sns
sns.set(color_codes=True)
from matplotlib.pyplot import *
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
np.set_printoptions(threshold=np.inf)
font = {'family': 'Microsoft YaHei',  'weight': 'bold',  'size': '16'}
matplotlib.rc("font", family="Microsoft YaHei", weight="bold", size="16")
df = pd.read_csv('./data/dataset03.csv')
X_train = df.iloc[:,1:]
df_test = pd.read_csv('./data/test_dataset.csv')
X_test =df_test.iloc[:,1:]
act_func = 'relu'
model = Sequential()
TIME_PERIODS = 44
input_shape=(TIME_PERIODS,)
model.add(Reshape((TIME_PERIODS,1), input_shape=input_shape))
model.add(Conv1D(64, 4, strides=2, activation='relu', input_shape=(TIME_PERIODS, 1)))
model.add(Conv1D(64,4, strides=2, activation='relu', padding="same"))
