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
from data_helper import *
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
look_back = int(config['model']['look_back'])
print("Loaded variables from config file...\n")
print(f"Using: {round(train_percent *100, 2)} % for training...")
print(f"Using: {round(validation_percent *100, 2)} % for validating...")
print(f"Iterating: {n_jobs} times...")
print(f"Epochs: {epochs}")
print(f"2 hidden layers with {neurons} neurons...")

fig_name = config['outfiles']['fig_name']
weight_name = config['outfiles']['weight_name']
arch_name = config['outfiles']['arch_name']
show_every = int(config['outfiles']['show_every'])

point_list = [point_name, 'aiTIT4045']
df = pc.get_stream_by_point(point_list, start = start, end = end, calculation = calculation, interval= interval)
df = df.dropna(how='any')


df =  clean_train_data(df, eval_expression=["df.loc[df['GBSF_Electricity_Demand_kBtu'] > 2400]"])
df = create_standard_multivariable_df(df, shift = look_back)


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

vertical_lines = [x for x in range(epochs) if x % show_every == 0]

plt.plot(train, color='blue', label='train')
plt.plot(val, color='orange', label='validation')
plt.title(f'LSTM model Train vs Validation loss\n 2 Layers- {neurons} Neurons')
plt.ylabel('loss (mse)')
plt.xlabel('epoch')
plt.legend()

unflat = train.values.tolist()
unflat1 = val.values.tolist()
flatten = [ item for sublist in  unflat for item in sublist]
flatten1 = [ item for sublist in  unflat1 for item in sublist]
for xc in vertical_lines:
    plt.axvline(x=xc, color = 'r', linestyle = '--')
for i,j in zip(train.index, train.values):
    if i % show_every == 0:
        plt.annotate(str(j), xy = (i,j ), xytext=(i+1, j+.000200),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
for i,j in zip(val.index, val.values):
    if i % show_every == 0:
        plt.annotate(str(j), xy = (i,j ), xytext=(i+1, j+.0010),arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

plt.savefig(fig_name)

# serialize model to JSON
model_json = history.model.to_json()
with open(arch_name, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(weight_name)
print("Saved model to disk")