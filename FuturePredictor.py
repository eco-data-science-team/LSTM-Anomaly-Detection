import sys
import pandas as pd
import configparser
import numpy as np
import matplotlib.pyplot as plt
#from datetime import datetime
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler

from data_helper import *

config = configparser.ConfigParser()
config.read('config/mypredictorconfig.ini')

eco_tools_path = config['SETUP']['eco_tools_path']
sys.path.append(eco_tools_path)
from ecotools.pi_client import pi_client
pc = pi_client(root = 'readonly')
point_name = config['PI']['point_name']
start = config['PI']['start']
end = config['PI']['end']
interval = config['PI']['interval']
calculation = config['PI']['calculation']
forecaster = config['PI']['forecaster']

look_back = int(config['MODEL']['look_back'])
anomaly_threshold = int(config['MODEL']['anomaly_threshold'])
forecast = config['MODEL']['forecast']

point_list = [point_name, 'aiTIT4045']
df = pc.get_stream_by_point(point_list, start = start, end = end, calculation = calculation, interval= interval)
df1 = pc.get_stream_by_point(forecaster, end = forecast, interval = interval, calculation = calculation)
new_df = pd.concat([df,df1], axis = 1, sort = False)


# Model reconstruction from JSON file
weight_name = config['infiles']['weight_name']
arch_name = config['infiles']['arch_name']
scaler = MinMaxScaler(feature_range=(0,1))
with open(arch_name, 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(weight_name)

del df, df1
#we fill the null values for outside air temp with Future TMY data
# then we drop the Outside_Air_Temp_Forecast column as it is not needed anymore
new_df['aiTIT4045'].fillna(new_df[forecaster], inplace = True)
new_df.drop(forecaster, axis = 1, inplace = True)

values_to_predict = new_df.loc[new_df[point_name].isna()].shape[0]
print(f"Values to Predict: {values_to_predict}")
def generate_prediction(df):
    y = df[point_name]
    X = df.drop(columns = point_name)
    X_test = scaler.fit_transform(X)
    y_test = scaler.fit_transform(np.array(y).reshape((-1,1)))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    prediction = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction.reshape(-1,1))
    index = df.index
    result = pd.DataFrame({"Actual":scaler.inverse_transform(y_test).reshape((-1,)),
                                   "Modeled":prediction.reshape((-1,))}, index=index)
    return result.tail(2)
    
firstNaN = new_df['aiTIT4045'].index.searchsorted(new_df[point_name].isna().idxmax())
for i in range(values_to_predict):
    print(f"Predicting: {i+1} / {values_to_predict}")
    #find the index of where the first nan appears in outside air temp
    nan_index = new_df['aiTIT4045'].index.searchsorted(new_df[point_name].isna().idxmax())
    #print(f"NaN index: {nan_index}")
    #create a DataFrame that has the tail of 10 points and has the first NaN value 
    #of point interested in as NaN
    df = new_df.iloc[ :nan_index + 1]#.tail(15)
    df = create_standard_multivariable_df(df, shift = look_back, dropna = False)
    #print(f"Rolling 24h mean: {df.Rolling24_mean.tail(2)}")
    #drop all NaN values except the very last one as this is the 
    #one we are interested in predicting then append that last row
    #from df onto ddf
    ddf = df.iloc[:-1,:].dropna()
    ddf = ddf.append(df.iloc[-1], verify_integrity = True)
    result = generate_prediction(ddf)
    new_df.iloc[nan_index].fillna(result.Modeled.iloc[-1], inplace = True)


#new_df[point_name].tail(values_to_predict +1).plot(figsize = (20,10))
new_df[point_name].plot(figsize = (20,10))
new_df[point_name].iloc[firstNaN:].plot(figsize = (20,10), color = 'r')
plt.axvline(x = new_df.iloc[firstNaN].name, color = 'green', linestyle = '--')
plt.suptitle(f"{point_name}\n", fontsize = 24)
plt.title(f"Predicting {forecast} using {forecaster}", fontsize = 20)
plt.savefig(f'Future_predicted_{forecaster}.png')

new_df.to_csv(f'{forecaster}_saved_predictions_{firstNaN}.csv')
