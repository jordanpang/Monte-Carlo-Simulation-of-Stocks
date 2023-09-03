import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override() #Getting the closing prices data

# import data
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end) #Collects stock data and puts into dataframe
    stockData = stockData['Close'] #Interested in close prices daily
    returns = stockData.pct_change() #Percentage changes
    meanReturns = returns.mean()
    covMatrix = returns.cov() #How the stocks vary with each other. >0 if move in same direction.
    return meanReturns, covMatrix

stockList = ['TLT', 'AEP', 'KO', 'PFE'] #US Treasury bonds, American Electrical Power, Coca Cola, Vanguard Real Estate, Pfizer
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

meanReturns, covMatrix = get_data(stocks, startDate, endDate)

weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

mc_sims = 100 #Number of simulations
T = 150 #Time in days

meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T #Transpose

portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)

initialPortfolio = 30000

for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))
    L = np.linalg.cholesky(covMatrix)
    dailyReturns = meanM + np.inner(L, Z)
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialPortfolio

plt.plot(portfolio_sims)
plt.ylabel('Porfolio Value ($)')
plt.xlabel('Days')
plt.title('Monte Carlo simulation of a stock portfolio')
plt.show()