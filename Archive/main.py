from matplotlib import style
from collections import Counter
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import os
import pickle
import getSP500
import numpy as np
import mplfinance as mpl
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

#df = pd.read_csv('pltr.csv', parse_dates=True, index_col=0)
#s = mpl.make_mpf_style(base_mpf_style = 'nightclouds', marketcolors = mpl.make_marketcolors(up='g',down='r',edge='inherit',wick='inherit'))
#mpl.plot(df, type = 'candlestick', mav = 10, volume = True, style = s)

def getYahooData(reloadSP500 = False):
    if reloadSP500:
        tickers = getSP500.saveSP500Tickers()
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2010,1,1)
    end = dt.datetime(2020,12,1)

    for ticker in tickers:
        print(ticker)
        ticker = ticker.replace('.', '-')

        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

#getYahooData()

def compileData():
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    mainDF = pd.DataFrame()

    for count,ticker in enumerate(tickers):
        ticker = ticker.replace('.', '-')
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace = True)

        df.rename(columns = {'Adj Close': ticker}, inplace = True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace = True)

        if mainDF.empty:
            mainDF = df
        else:
            mainDF = mainDF.join(df, how = 'outer')

        if count % 10 == 0:
            print(count)

    print(mainDF.head())
    mainDF.to_csv('sp500Joined.csv')

def visualizeData():
    df = pd.read_csv('sp500Joined.csv')
    # df['AAPL'].plot()
    # plt.show()
    dfCorr = df.corr()

    print(dfCorr.head())

    data = dfCorr.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    heatmap = ax.pcolor(data, cmap = plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor = False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor = False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = dfCorr.columns
    row_labels = dfCorr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation = 90)
    heatmap.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()

def processDataForLabels(ticker):
    days = 7
    df = pd.read_csv('sp500Joined.csv', index_col = 0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace = True)

    for i in range(1, days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0, inplace = True)
    return tickers, df

def buySellHold(*args):
    cols = [c for c in args]
    req = 0.025

    for col in cols:
        if col > 0.03:
            return 1
        if col < -0.026:
            return -1

    return 0

def extractFeaturesets(ticker):
    tickers, df = processDataForLabels(ticker)

    df['{}_target'.format(ticker)] = list(map(buySellHold,
                                                df['{}_1d'.format(ticker)],
                                                df['{}_2d'.format(ticker)],
                                                df['{}_3d'.format(ticker)],
                                                df['{}_4d'.format(ticker)],
                                                df['{}_5d'.format(ticker)],
                                                df['{}_6d'.format(ticker)],
                                                df['{}_7d'.format(ticker)]))
    
    vals = df['{}_target'.format(ticker)].values.tolist()
    strVals = [str(i) for i in vals]
    print('Data spread: ', Counter(strVals))

    df.fillna(0, inplace = True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace = True)

    dfVals = df[[ticker for ticker in tickers]].pct_change()
    dfVals = dfVals.replace([np.inf, -np.inf], 0)
    dfVals.fillna(0, inplace = True)

    X = dfVals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df

def doML(ticker):
    X, y, df = extractFeaturesets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('Accuracy', confidence)
    predictions = clf.predict(X_test)
    print('Predicted spread: ', Counter(predictions))

    return confidence

doML('BAC')