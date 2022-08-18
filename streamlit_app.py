
    #Importing Libraries
import pandas as pd
import numpy as np
import streamlit as st
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_datareader as data
import flask as fsk
from datetime import date
import yfinance as yf
from chart_studio.plotly import plot, iplot, plotly
#import plotly.plotly as py

        # from fbprophet import Prophet
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed 
set_random_seed(0)
#from neuralprophet.plot import plot_plotly
#from plotly import graph_objs as go
import plotly.express as px
   #import plotly.graph_objects as go 
#import plotly.plotly as py
startdate = '2010-01-01'
enddate = date.today().strftime("%Y-%m-%d")
#enddate = datetime.date.today()
st.title("Stock Prediction app for long duration")
   
stocks = ("AAPL", "GOOG", "HDFCBANK.NS", "^NSEI","^NSEBANK")
selected_stock = st.selectbox("Select Dataset for prediction", stocks)
## n_years = st.slider("prediction Months:", 1, 3)
n_weeks = st.slider("prediction Days:", 1, 7)
period = n_weeks * 24

@st.cache
def load_data(ticker):
    data = yf.download(tickers='HDFCBANK.NS', period='1d', interval='5m')
#    data = yf.download(ticker,startdate, enddate)
 #   data.date = data.date.strftime("%Y-%m-%d")
 #   data[Datetime]= data.Datetime.strftime("%Y-%m-%d")
    data.reset_index(inplace = True)
    return data

#data_load_state = st.text("load data")
data = load_data(selected_stock)
#data_load_state.text("data loading....done!")

st.subheader('Raw data')
st.write(data.tail())

#st.write(data.Date)

#fig = go.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
#fig.show()

#def plot_raw_data():
#import plotly.graph_objects as go
#fig = go.Figure()
#fig = go.scatter(x = data['Close'], y = data['Open'], name ='Open price')
#fig = go.scatter(x = data['Close'], y = data['Open'], name ='Open price')
    #fig.add_trace(go.Scatter(x=data['Date'], y= data['Open'], name ='Open price')
   # fig.add_trace(go.Scatter(x=data['Date'], y= data['Close'], name ='Close price')
#fig.layout.update(title_text("time Series Data", xaxis_rangeslider_visible =True)
   # st.plotly_chart(fig)
   # plot_raw_data()
                  
#fig.show()

# Prediction .. ... 
df_train = data[['Datetime', 'Close']]
df_train = df_train.rename(columns= {"Datetime": "ds", "Close": "y"})

m = NeuralProphet()
m.fit(df_train, freq = "H")
future = m.make_future_dataframe(df_train, periods = period)
forecast = m.predict(future)
st.subheader('Forecast Data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

