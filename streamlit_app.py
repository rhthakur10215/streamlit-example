import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import datetime

import pandas_datareader as data
import flask as fsk
import yfinance as yf

from datetime import date
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.models import load_model

import streamlit as st
# To remove the scientific notation from numpy arrays
np.set_printoptions(suppress=True)

from nsepy import get_history
from datetime import datetime
# install the nsepy library to get stock prices
#!pip install nsepy
#######################################
start = '2018-01-01'
end = '2022-05-22'
############################################
# Getting Stock data using nsepy library


#########################################
#endDate=date.today().strftime("%Y-%m-%d")
# Fetching the data
#StockData=get_history(symbol='INFY', start=startDate, end=endDate)
#StockData = get_history(symbol="NIFTY", start=startDate, end=endDate, index=True)  
#StockData = get_history(symbol="BANKNIFTY", start=startDate, end=endDate, index=True)                   
                      
#end = date.today()
#df = data.DataReader('HDFCBANK.NS','yahoo', start, end)

#df = data.DataReader('HDFCBANK.NS','yahoo', start, end)
user_input = ['HDFCBANK.NS']
#def UpdateCSV(stock = 'HDFCBANK.NS', update = True ):
df= yf.download(user_input,'2016-01-01','2022-05-22')
#df = get_history(symbol="BANKNIFTY", start=start, end=end, index=True) 
#assert None == df.to_csv("DATA22.csv")
#if 0:
#    UpdateCSV()

##########################
st.title('Stock Trend Prediction')
#df = data.DataReader('HDFCBANK.NS','yahoo', start, end)
#df = data.get_data_yahoo('HDFCBANK.NS',start='2010-01-01', end='2022-01-01')
user_input = st.text_input('Enter ticker name','HDFCBANK.NS')
#df = data.DataReader('user_input','yahoo', start, end)
#df= yf.download('user_input','2016-01-01','2022-01-23')
df= yf.download(user_input,'2016-01-01','2022-05-22')
#df = get_history(symbol="BANKNIFTY", start=start, end=end, index=True) 
st.subheader('Data from live market')
st.write(df.describe())

st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize=(16,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart of 100MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig = plt.figure(figsize=(16,6))
plt.plot(ma100)
plt.plot(df.Close)
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart of 100MA & 200Ma')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig = plt.figure(figsize=(16,6))
plt.plot(ma100, 'r', label = '100 Moving avg')
plt.plot(ma200, 'g', label = '200 Moving avg')
plt.plot(df.Close, 'b', label = 'Actual Price')
plt.legend()
st.pyplot(fig)
                 
#Data training and testing    
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])    

#Importing MinMax Scaler :
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training) #data_training

x_train = []
y_train = []
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units=50, activation = 'relu', return_sequences = True, 
               input_shape =(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

# Load my ready model
model = load_model('keras_model.h5')

#testing Part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

#Prediction Section
y_predicted = model.predict(x_test)
div = scaler.scale_
scale_factor = 1/div[0]
#scale_factor = 1/0.00108578

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader(' Final grapg of prected values Vs actual price')
fig2 = plt.figure(figsize = (16,6))
plt.plot(y_test, 'b', label = 'Original price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
    
