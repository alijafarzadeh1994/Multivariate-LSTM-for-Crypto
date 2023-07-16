import yfinance as yf
from ta import add_all_ta_features
from ta.utils import dropna
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.momentum import StochasticOscillator
import pandas as pd

# Download Bitcoin data
data = yf.download('BTC-USD', start='2022-01-01', end='2023-07-06')

# Clean the data by removing any rows with missing values
data = dropna(data)

# Add RSI indicator
rsi = RSIIndicator(close=data['Close']).rsi()
data['RSI'] = rsi

# Calculate MACD
macd = MACD(close=data['Close'], window_slow=26, window_fast=12, window_sign=9)
data['MACD'] = macd.macd()
data['MACD_Signal'] = macd.macd_signal()
data['MACD_Histogram'] = macd.macd_diff()

# Calculate Stochastic Oscillator
stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
data['%K'] = stoch.stoch()
data['%D'] = stoch.stoch_signal()

# Select the required columns: Close, RSI, MACD, MACD Signal, MACD Histogram, %K, %D
data.fillna(0, inplace=True)  # Fill null values with zero
data = data.iloc[3:453,:]
selected_data = data[[ 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', '%K', '%D']]

# Save the selected data to a CSV file
selected_data.to_csv("btc_indicators.csv", index=False)





df = pd.read_csv('dftry01.csv',index_col = 1)
df = df.iloc[:,1:]
df = df.rename_axis('date')
df3 = selected_data
# Assuming 'Date' column is the index of the DataFrame df3
df3.index = pd.to_datetime(df3.index)
df3['new_date'] = df3.index.strftime("%-m/%-d/%Y")
df3.set_index('new_date', inplace=True)
df3 = df3.iloc[:,:6]
df3 = df3.rename_axis('date')
dataset = pd.concat([df, df3], axis = 1)
dataset
dataset.to_csv('dataset.csv')




