# Import libraries
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.stats import norm
import yfinance as yf
import seaborn as sns

style.use('seaborn')

# yfinance function override
''' Function that its used to override the functionality of pandas-datareader and allow the yfinance library 
 to be used instead to access Yahoo Finance data. '''
yf.pdr_override()

# Ticker
ticker = 'AMZN'
# Function to dowload the data
def market_data(ticker):
   return wb.get_data_yahoo(ticker, '2015-1-1')['Adj Close']
data = market_data(ticker)

log_returns = np.log(1 + data.pct_change())
mu = log_returns.mean()
var = log_returns.var()
drift = mu - (0.5 * var)
stdev = log_returns.std()

days = 100
trials = 10000

# Z = norm.ppf(np.random.rand(days, trials))
# daily_returns = np.exp(drift + stdev * Z)
# price_line = np.zeros_like(daily_returns)
# price_line[0] = data.iloc[-1]

# for t in range(1, days):
#     price_line[t] = price_line[t-1] * daily_returns[t]

# plt.figure(figsize=(15,6))
# plt.plot(pd.DataFrame(price_line))
# plt.xlabel('Days')
# plt.ylabel(f'{ticker} Price')
# sns.displot(pd.DataFrame(price_line).iloc[-1])
# plt.xlabel(f'Price at {str(days)} days')
# plt.ylabel('Frequency')
# plt.show()

