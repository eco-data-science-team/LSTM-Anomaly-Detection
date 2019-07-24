import sys
import pandas as pd
import configparser
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
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

# Model reconstruction from JSON file
weight_name = config['infiles']['weight_name']
arch_name = config['infiles']['arch_name']
fig_name = config['outfiles']['fig_name']
with open(arch_name, 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(weight_name)

look_back = int(config['MODEL']['look_back'])
anomaly_threshold = int(config['MODEL']['anomaly_threshold'])

point_list = [point_name, 'aiTIT4045']
df = pc.get_stream_by_point(point_list, 
start = start, end = end, calculation = calculation, interval= interval)

df = create_standard_multivariable_df(df)

scaler = MinMaxScaler(feature_range = (0,1))
def prep_prediction_data(df):
    y = df[point_name]
    X = df.drop(columns = point_name)
    X_ = scaler.fit_transform(X)
    y_ = scaler.fit_transform(np.array(y). reshape((-1,1)))
    X_ = np.reshape(X_, (X_.shape[0], 1 , X_.shape[1]))
    
    return X_, y_

X, y = prep_prediction_data(df)
prediction =  model.predict(X)
prediction = scaler.inverse_transform(prediction.reshape(-1, 1))

index = df.index

result = pd.DataFrame({"Actual":scaler.inverse_transform(y).reshape((-1,)),"Modeled":prediction.reshape((-1,))}, index=index)

result.eval('Difference = (Actual - Modeled)/ Modeled * 100', inplace=True)

result["Difference"] = result['Difference'].abs().round(decimals = 2)                                 

actual = result['Actual'].tolist()
modeled = result['Modeled'].tolist()
difference = result['Difference'].tolist()
idx = result.index.tolist()
ymax = max(max(actual, modeled))
ymin = min(min(actual, modeled))

plt.figure(figsize=(18,10))
count = 1
for ii in range(len(actual)):
    
    if difference[ii] > anomaly_threshold:
        if count%2 == 0:
            plt.text(idx[ii] , ymin*0.99, int(difference[ii]), size = 11)
            count = count +1
        else:
            plt.text(idx[ii] , ymin*1.00, int(difference[ii]), size = 11)
            count = count +1
        plt.axvline(x = idx[ii], color = 'r', linestyle = '--')
plt.plot(idx, actual, marker = ".", color="#5bc0de", label = 'Actual')
plt.plot(idx, modeled, marker = ".", color="#E8743B", label = "Modeled")

plt.ylim([ymin*0.985, ymax*1.01])
plt.fill_between(idx, actual, modeled, color = "grey", alpha = "0.3")
plt.yticks(actual, size= 10)
plt.xticks(idx, size = 10)

plt.locator_params(axis = 'y', tight = True, nbins=6)
plt.locator_params(axis = 'x', nbins = 6)
biggest_difference = result.loc[result['Difference'] == max(difference)]['Difference'][0]
at_time = result.loc[result['Difference'] == max(difference)].index[0]
plt.suptitle(f"{point_name}\n Actual vs Modeled", fontsize = 16)
plt.title(f"Threshold: {anomaly_threshold}%\n Biggest Difference:{int(biggest_difference)}% on {at_time}")
plt.legend()
plt.savefig(fig_name)
#plt.show()