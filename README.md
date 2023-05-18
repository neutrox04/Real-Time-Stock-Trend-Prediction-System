# Real-Time Stock Prediction System

This is a real-time stock prediction system that utilizes machine learning and technical analysis to predict the future trends of stock prices. The system is based on the code provided in the repository.

## Features

- Stock data retrieval: The system retrieves stock data from Yahoo Finance using the `yfinance` library. Users can input a stock ticker symbol and select a start date to fetch the historical stock data.

- Bollinger Bands: The system calculates and plots Bollinger Bands, which are used to analyze volatility and potential price breakouts.

- Price vs Time Charts: The system provides various charts to visualize the closing price of the stock over time. Users can view the closing price vs time, closing price with a 100-day moving average, and closing price with both 100-day and 200-day moving averages.

- Training and Testing: The system splits the stock data into training and testing sets for model evaluation. It uses a MinMaxScaler to scale the data before feeding it into the machine learning model.

- Model Loading and Testing: The system loads a pre-trained machine learning model (`adani.h5`) and performs predictions on the test data. It calculates the Root Mean Squared Error (RMSE) as an evaluation metric.

- Candlestick Chart: The system generates a candlestick chart using the `plotly` library to visualize the stock's open, high, low, and close prices.

- Scatter Plot: The system creates a scatter plot to analyze the relationship between the stock's closing price and the volume traded.

- Relative Strength Index (RSI): The system calculates and plots the RSI, a momentum indicator used to identify overbought and oversold conditions.

- Moving Average Convergence Divergence (MACD): The system computes and visualizes the MACD, a trend-following momentum indicator.


-Dashboard
![image](https://github.com/neutrox04/Real-Time-Stock-Trend-Prediction-System/assets/87473552/2189f5f7-eaa0-4e6c-8fd5-02c6c8c6a2ac)
-Closing Price vs Time Chart
![image](https://github.com/neutrox04/Real-Time-Stock-Trend-Prediction-System/assets/87473552/37ff471d-c6be-4425-b9d5-e2b19354eb8e)
-100 & 200 days Moving average
![image](https://github.com/neutrox04/Real-Time-Stock-Trend-Prediction-System/assets/87473552/dff04a48-22eb-458e-874f-5d923cd84d6b)
-Predictions vs Original chart
![image](https://github.com/neutrox04/Real-Time-Stock-Trend-Prediction-System/assets/87473552/c40e9642-13f7-4629-9813-abeec77ada2c)
-Candlestick Chart
![image](https://github.com/neutrox04/Real-Time-Stock-Trend-Prediction-System/assets/87473552/928a1331-ae9c-4489-9217-53a9065e9fee)
-Scatter Plot
![image](https://github.com/neutrox04/Real-Time-Stock-Trend-Prediction-System/assets/87473552/a35c95e3-8026-4381-8959-fea45eab2642)
-Relative Strength Index
![image](https://github.com/neutrox04/Real-Time-Stock-Trend-Prediction-System/assets/87473552/c4f4a6a5-1edc-4789-8f0c-7126c6126ddb)
-Moving Average Convergence Divergence

## Dependencies

- `numpy`: Numerical computing library.
- `pandas`: Data manipulation library.
- `matplotlib`: Plotting library.
- `yfinance`: Library for retrieving stock data from Yahoo Finance.
- `keras`: Deep learning library for loading and using the pre-trained model.
- `streamlit`: Library for building interactive web applications.
- `scikit-learn`: Machine learning library for scaling the data and calculating the RMSE.
- `plotly`: Library for creating interactive plots and charts.
- `mplfinance`: Library for plotting candlestick charts.
- `ta`: Library for technical analysis indicators.

## Installation and Usage

1. Clone the repository:

```
git clone <repository_url>
```

2. Install the dependencies using `pip` or `conda`:

```
pip install -r requirements.txt
```

or

```
conda env create -f environment.yml
conda activate stock-prediction
```

3. Run the application:

```
streamlit run stock_prediction.py
```

4. Open your web browser and go to `http://localhost:8501` to access the real-time stock prediction system.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or improvements, please open an issue or submit a pull request.

Feel free to customize the README according to your specific requirements and add any additional sections or information as needed.
