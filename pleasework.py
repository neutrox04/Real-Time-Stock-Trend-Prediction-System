import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
import mplfinance as mpf
# Importing necessary libraries
import ta
from datetime import datetime
from urllib.request import urlopen

st.markdown("<h1 style='text-align: center;'>Real-Time Stock Trend Prediction</h1>", unsafe_allow_html=True)
# Input fields for stock ticker and start date
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
start_date = st.date_input('Enter Start Date', datetime(2018,1,1))

# Set end date as current date
end_date = datetime.today().strftime('%Y-%m-%d')

# Download stock data from Yahoo Finance
df = yf.download(user_input, start_date.strftime('%Y-%m-%d'), end_date)

stock_data = yf.download(user_input, start=start_date, end=None)

# Calculate Bollinger Bands
indicator_bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)

# Add Bollinger Bands to DataFrame
df['bb_high'] = indicator_bb.bollinger_hband()
df['bb_low'] = indicator_bb.bollinger_lband()

# Plot Bollinger Bands
fig, ax = plt.subplots(figsize=(12,6))
df['Close'].plot(ax=ax, label='Price')
df['bb_high'].plot(ax=ax, label='Upper Band')
df['bb_low'].plot(ax=ax, label='Lower Band')
plt.title(f'{user_input} Bollinger Bands')
plt.legend()
plt.show()
# Describing Data
st.subheader('Data from {} to {}'.format(start_date.strftime('%Y-%m-%d'), end_date))
st.write(df.describe())




#Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

#Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

data_training_array = scaler.fit_transform(data_training)

#Load my model

model = load_model('adani.h5')

#Testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test =[]
y_test =[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))

# Display RMSE value
st.subheader('Root Mean Squared Error')
st.write('%.2f' % rmse)

# Explain RMSE
explanation = "The RMSE value is a measure of how much the predicted stock prices deviate from the actual stock prices on average. In this case, the RMSE value of **" + str(round(rmse, 2)) + "** means that, on average, the predicted stock prices deviated from the actual stock prices by approximately **" + str(round(rmse, 2)) + "** dollars. Lower values of RMSE indicate that the predictions are more accurate."
st.info(explanation)

#Final Graph
st.markdown("<h1 style='text-align: center;'>Predictions vs Original</h1>", unsafe_allow_html=True)
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)



# Define figure
fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

st.markdown("<h1 style='text-align: center;'>Stock Price Analysis with Candlestick Chart</h1>", unsafe_allow_html=True)
# Add title and axis labels
fig.update_layout(
    title='Candlestick Chart for {}'.format(user_input),
    yaxis_title='Price',
    xaxis_title='Date'
)

# Display figure
st.plotly_chart(fig)



# scatter plot
st.markdown("<h1 style='text-align: center;'>Closing Price vs Volume Traded Scatter Plot</h1>", unsafe_allow_html=True)
# Explanation for scatter plot
st.markdown("### Scatter Plot: ")
explanation = "A scatter plot can be used to visualize the relationship between two variables, like the price of a stock and the volume of shares traded. This can help users identify patterns or trends in the data."
st.info(explanation)

fig3 = plt.figure(figsize=(12,6))
plt.scatter(df.Volume, df.Close, alpha=0.5)
plt.title('{} Closing Price vs Volume Traded Scatter Plot'.format(user_input))
plt.xlabel('Volume Traded')
plt.ylabel('Closing Price')
st.pyplot(fig3)


# Compute RSI
delta = df['Close'].diff()
gain = delta.mask(delta < 0, 0)
loss = -delta.mask(delta > 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean().abs()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

# Plot RSI
st.markdown("<h1 style='text-align: center;'>Relative Strength Index</h1>", unsafe_allow_html=True)

fig3 = plt.figure(figsize=(12,6))
plt.plot(df.index, rsi, 'purple', label = 'RSI')
plt.axhline(30, linestyle='--', color='orange')
plt.axhline(70, linestyle='--', color='orange')
plt.fill_between(df.index, y1=30, y2=rsi, alpha=0.1, color='green')
plt.fill_between(df.index, y1=70, y2=rsi, alpha=0.1, color='red')
plt.xlabel('Time')
plt.ylabel('RSI')
plt.legend()
st.pyplot(fig3)

# Explanation

explanation = "**RSI (Relative Strength Index):** The RSI is a momentum indicator that measures the strength of a stock\'s price action. It oscillates between 0 and 100, with readings above 70 indicating overbought conditions and readings below 30 indicating oversold conditions. Traders often use RSI to identify potential trend reversals and to confirm the strength of a current trend."
st.info(explanation)

# Calculate MACD and signal line
macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
df['MACD'] = macd.macd()
df['Signal Line'] = macd.macd_signal()

# Plot MACD and signal line
st.markdown("<h1 style='text-align: center;'>Moving Average Convergence Divergence (MACD)</h1>", unsafe_allow_html=True)
st.line_chart(df[['MACD', 'Signal Line']])

#Explanation
explanation = "The **MACD** is a popular technical indicator that is used to identify changes in momentum and trend in a stock. It is calculated by subtracting the 26-day exponential moving average (EMA) from the 12-day EMA. A 9-day EMA of the MACD, called the **signal line**, is then plotted on top of the MACD line, which can be used as a trigger for buy and sell signals. When the MACD line crosses above the signal line, it is considered a bullish signal, and when it crosses below the signal line, it is considered a bearish signal. The MACD histogram, which is the difference between the MACD line and the signal line, can also be used to identify changes in momentum and trend."
st.info(explanation)
