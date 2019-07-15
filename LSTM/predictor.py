import sys
import pandas as pd
import configparser
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta

from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
config = configparser.ConfigParser()
config.read('predictorconfig.ini')

eco_tools_path = config['SETUP']['eco_tools_path']
sys.path.append(eco_tools_path)
from ecotools.pi_client import pi_client

# Model reconstruction from JSON file
weight_name = config['infiles']['weight_name']
arch_name = config['infiles']['arch_name']
with open(arch_name, 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(weight_name)


import warnings
warnings.filterwarnings('ignore')
scaler = MinMaxScaler(feature_range=(0,1))

pc = pi_client(root = 'readonly')


point_name = config['PI']['point_name']

def create_dataset(dataset, look_back = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i: (i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)

def predict_and_transform(trainX):
    prediction = model.predict(trainX)
    return scaler.inverse_transform(prediction.reshape(-1,1))

def get_error(df):
    df['Error_Per'] = abs((df['Actual'] - df['Predicted']))/df['Actual'] * 100
    df['Error_Per'] = df['Error_Per'].replace(np.inf, 100)
    return df


def prep_df(df):
    df = df.dropna(how='any')
    dataset = df.values
    dataset = dataset.astype('float64')
    dataset = scaler.fit_transform(dataset)
    trainX, trainY = create_dataset(dataset, look_back = 3)
    trainX = np.reshape(trainX, (trainX.shape[0],1,  trainX.shape[1]))
    actual = scaler.inverse_transform(trainY.reshape(-1,1))
    prediction = predict_and_transform(trainX)
    actual = actual.reshape((-1,))
    prediction = prediction.reshape((-1,))
    df = pd.DataFrame({'Predicted': prediction, 'Actual': actual}, 
                    index = df.index[-len(prediction)-1:-1], 
                    columns =["Predicted", "Actual"])
    df = get_error(df)

    return df





df = pc.get_stream_by_point(point_name, end = '*')

df = prep_df(df)

print(df.tail())