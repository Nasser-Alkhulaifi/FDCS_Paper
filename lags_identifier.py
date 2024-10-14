import pandas as pd

def find_top_correlated_lags(df, columns, lag_range):
    """
    Identify the top correlated lags for specified columns in a DataFrame.
    This function calculates the correlation of each specified column with its lagged versions
    within a given range. It then identifies the top 10 lags with the highest absolute correlation
    for each column.
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (list of str): The list of column names to analyze.
    lag_range (tuple of int): A tuple specifying the range of lags to consider (start, end).
    Returns:
    dict: A dictionary where keys are column names and values are lists of the top 10 lags
          with the highest absolute correlation for each column.
    """
    top_lags = {}
    for column in columns:
        correlations = []
        for lag in range(lag_range[0], lag_range[1] + 1):
            lagged_series = df[column].shift(periods=lag)
            correlation = df[column].corr(lagged_series)
            correlations.append((lag, correlation))
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        top_lags[column] = [lag for lag, corr in correlations[:10]]

    return top_lags
