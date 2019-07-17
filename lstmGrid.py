# 1- Download data to train on
# 2- Clean up data (drop NaNs)
# 3- Add metrics to LSTM
import sys
import pandas as pd
#import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import configparser



#import keras.backend as K
#K.tensorflow_backend._get_available_gpus()
import multiprocessing
import keras
from keras.models import Sequential
from keras.optimizers import *
from keras.layers import LSTM, Embedding, Dense
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error




import warnings
warnings.filterwarnings('ignore')

np.random.seed(7)
callbacks_list = [EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01, verbose=2)]

scaler = MinMaxScaler(feature_range=(0,1))
config = configparser.ConfigParser()
config.read('config/lstmconfig.ini')

eco_tools_path = config['SETUP']['eco_tools_path']
sys.path.append(eco_tools_path)
from ecotools.pi_client import pi_client
pc = pi_client(root = 'readonly')

point_name = config['PI']['point_name']
start = config['PI']['start']
end = config['PI']['end']
interval = config['PI']['interval']
calculation = config['PI']['calculation']

train_percent = float(config['model']['train_percent'])
look_back = int(config['model']['look_back'])
n_jobs = int(config['model']['n_jobs'])
#n_jobs = multiprocessing.cpu_count() - 2


weight_name = config['outfiles']['weight_name']
arch_name = config['outfiles']['arch_name']
grid_obj = config['outfiles']['grid_object']
print("Loaded variables from config file...")

df = pc.get_stream_by_point(point_name, start = start, end = end, calculation = calculation, interval= interval)
df = df.dropna(how='any')
dataset = df.values
dataset = dataset.astype('float32')
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * train_percent)
test_size = len(dataset) - train_size
train, test = dataset[0 : train_size, :], dataset[train_size: len(dataset), :]
train_len = len(train)
test_len = len(test)
print(f"Training Length: {train_len} \n Test Length: {test_len}")
print(f"n_jobs = {n_jobs}")

def create_dataset(dataset, look_back = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i: (i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)

def create_model(neurons = 1):
    model = Sequential()
    model.add(LSTM(neurons, input_shape = (testX.shape[1], testX.shape[2]), return_sequences = True))
    model.add(LSTM(neurons))
    model.add(Dense(1, kernel_regularizer=keras.regularizers.l2(0.01),
                activity_regularizer=keras.regularizers.l1(0.01)))
    model.compile(optimizer = Adam(lr = 0.001), loss = 'mean_squared_error')

    return model
import itertools
import os


#look_back = 3
trainX, trainY = create_dataset(dataset, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0],1,  trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = KerasRegressor(build_fn= create_model, verbose = 2, batch_size = 16)
K.clear_session()

batch_size = [16, 32]
epochs = [50, 100]
neurons = [10, 50, 100]
#callbacks = [callbacks_list]
#param_grid = dict(batch_size = batch_size, epochs = epochs, neurons = neurons, callbacks= callbacks)
param_grid = dict(batch_size = batch_size, epochs = epochs, neurons = neurons)
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, verbose = 3, n_jobs = n_jobs)

keras_grid = grid.fit(trainX, trainY, validation_split = 0.3)
from sklearn.externals import joblib
joblib.dump(keras_grid, grid_obj)
best_model = keras_grid.best_estimator_
best_model.model.save_weights(weight_name)
with open(arch_name, 'w') as f:
    f.write(best_model.model.to_json())

# summarize results
print("Best: %f using %s" % (keras_grid.best_score_, keras_grid.best_params_))
means = keras_grid.cv_results_['mean_test_score']
stds = keras_grid.cv_results_['std_test_score']
params = keras_grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))