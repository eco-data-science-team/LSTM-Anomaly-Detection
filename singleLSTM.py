import sys

import keras
from keras.optimizers import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt
from numpy import array
from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib



import configparser
config = configparser.ConfigParser()
config.read('config/mylstmconfig.ini')

scaler = MinMaxScaler(feature_range=(0,1))
eco_tools_path = config['SETUP']['eco_tools_path']
sys.path.append(eco_tools_path)
from ecotools.pi_client import pi_client
pc = pi_client(root = 'readonly')

#callbacks_list = [EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, verbose=2)]


point_name = config['PI']['point_name']
start = config['PI']['start']
end = config['PI']['end']
interval = config['PI']['interval']
calculation = config['PI']['calculation']

train_percent = float(config['model']['train_percent'])
validation_percent = 1.0 - train_percent
n_jobs = int(config['model']['n_jobs'])
epochs = int(config['model']['epochs'])
neurons = int(config['model']['neurons'])
print("Loaded variables from config file...\n")
print(f"Using: {round(train_percent *100, 2)} % for training...")
print(f"Using: {round(validation_percent *100, 2)} % for validating...")
print(f"Iterating: {n_jobs} times...")
print(f"Epochs: {epochs}")
print(f"2 hidden layers with {neurons} neurons...")

fig_name = config['outfiles']['fig_name']
weight_name = config['outfiles']['weight_name']
arch_name = config['outfiles']['arch_name']

def create_mulitvariable_df(data):
    data.rename(columns={'aiTIT4045':'OAT'}, inplace=True)

    data["cdd"] = data.OAT - 65.0
    data.loc[data.cdd < 0, "cdd"] = 0
    data["hdd"] = 65.0 - data.OAT
    data.loc[data.hdd < 0, 'hdd'] = 0
    data["cdd2"] = data.cdd**2
    data["hdd2"] = data.hdd**2

    data2 = data.copy()
    del data
    month = [str('MONTH_'+ str(x+1)) for x in range(12)]
    data2["MONTH"]= data2.index.month
    data2["MONTH"] = data2["MONTH"].astype('category')
    month_df = pd.get_dummies(data=data2, columns=['MONTH'])
    
    month_df = month_df.T.reindex(month).T.fillna(0)
    month_df = month_df.drop(month_df.columns[0], axis = 1)
    
    tod = [str('TOD_' + str(x)) for x in range(24)]
    data2["TOD"] = data2.index.hour
    data2["TOD"] = data2["TOD"].astype('category')
    tod_df = pd.get_dummies(data = data2, columns = ['TOD'])
    tod_df = tod_df.T.reindex(tod).T.fillna(0)
    tod_df = tod_df.drop(tod_df.columns[0], axis = 1)
    
    dow = [str('DOW_' + str(x)) for x in range(7)]
    data2["DOW"] = data2.index.weekday
    data2["DOW"] = data2["DOW"].astype('category')
    dow_df = pd.get_dummies(data = data2, columns = ['DOW'])
    dow_df = dow_df.T.reindex(dow).T.fillna(0)
    dow_df = dow_df.drop(dow_df.columns[0], axis = 1)
    
    ### Create Weekend flag
    data2["WEEKEND"] = 0
    data2.loc[(dow_df.DOW_5 == 1) | (dow_df.DOW_6 == 1), 'WEEKEND'] = 1
    
    data2["shift1"] = data2.iloc[:,0].shift(2)

    data2["rolling24_mean"] = data2.iloc[:,0].rolling('24h').mean()
    data2["rolling24_max"] = data2.iloc[:,0].rolling('24h').max()
    data2["rolling24_min"] = data2.iloc[:,0].rolling('24h').min()
    
    data2 = pd.concat([data2, month_df, tod_df, dow_df], axis =1)

    data2.dropna(inplace=True)
    
    return data2

point_list = [point_name, 'aiTIT4045']
df = pc.get_stream_by_point(point_list, start = start, end = end, calculation = calculation, interval= interval)
df = df.dropna(how='any')


def clean_train_data(df):
    #mask1 = (df[point_name] > 2400 )& (df.index.year < 2019)
    mask1 = (df[point_name] > 2400 )
    df1 = df.loc[mask1]
    #mask2 = (df.index.year>=2019)
    #df2 = df.loc[mask2]
    return df1

df = clean_train_data(df)
df = create_mulitvariable_df(df)


def scale_keras(X, y):
    # normalize the dataset
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler((0, 1))
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(np.array(y).reshape((-1,1)))
    # split into train and test sets
    train_size = int(len(X) * 0.7)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[test_size:len(X)]
    y_train, y_test = y[0:train_size], y[test_size:len(y)]
    return X_train, X_test, y_train, y_test, scaler_x, scaler_y

y = df[point_name]
X = df.drop(columns=point_name)
X_train, X_test, y_train, y_test, scaler_x, scaler_y = scale_keras(X, y)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


train = DataFrame()
val = DataFrame()
np.random.seed(42)
for i in range(n_jobs):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences = True))
    model.add(LSTM(neurons))
    model.add(Dense(1))
    model.compile(optimizer = Adam(lr = 0.001), loss = 'mean_squared_error')
    # fit model
    history = model.fit(X_train, y_train, epochs = epochs, validation_split = validation_percent, shuffle = False)
    # store history
    train[str(i)] = history.history['loss']
    val[str(i)] = history.history['val_loss']

from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(train, color='blue', label='train')
plt.plot(val, color='orange', label='validation')
plt.title('model train vs validation loss\n 2 Layers-100 Neurons')
plt.ylabel('loss (mse)')
plt.xlabel('epoch')
plt.legend()
plt.savefig(fig_name)

# serialize model to JSON
model_json = history.model.to_json()
with open(arch_name, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(weight_name)
print("Saved model to disk")