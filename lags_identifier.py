import pandas as pd

def find_top_correlated_lags(df, columns, lag_range):
    top_lags = {}
    for column in columns:
        correlations = []
        for lag in range(lag_range[0], lag_range[1] + 1):
            lagged_series = df[column].shift(periods=lag)
            correlation = df[column].corr(lagged_series)
            correlations.append((lag, correlation))
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        top_lags[column] = [lag for lag, corr in correlations[:10]]
    ## new tes
    return top_lags
