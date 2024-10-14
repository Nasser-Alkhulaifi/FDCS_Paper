from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from feature_engine.timeseries.forecasting import LagFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import CyclicalFeatures
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

target = 'Electricity Consumption'

# Custom transformer for calculating rolling window statistics
class RollingWindowFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, window=24, shift_period=169):
        self.window = window
        self.shift_period = shift_period
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        rolling_windows = X[target].rolling(window=self.window)
        rolling_features = pd.DataFrame({
            f'{target}_Mean_192': rolling_windows.mean().shift(self.shift_period),
            f'{target}_Std_192': rolling_windows.std().shift(self.shift_period),
            f'{target}_Max_192': rolling_windows.max().shift(self.shift_period),
            f'{target}_Min_192': rolling_windows.min().shift(self.shift_period),
            f'{target}_Kurt_192': rolling_windows.kurt().shift(self.shift_period),
            f'{target}_Skew_192': rolling_windows.skew().shift(self.shift_period)
        })
        return pd.concat([X, rolling_features], axis=1).fillna(0)

# Pipeline for preprocessing data
def preprocess_data(df):
    """
    Preprocesses the input DataFrame by performing several steps:
    - Converts 'DateTime' to datetime format and sets it as the index.
    - Adds 'Is_Open' column based on predefined working hours.
    - Splits the data into training, validation, and test sets.
    - Applies datetime features, cyclical features, lag features, and rolling window features.
    - Normalises the features and target variables using MinMaxScaler.
    - Saves the target scaler to a file.
    """
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M')
    df.set_index('DateTime', inplace=True)

    working_hours = {
        'Saturday': ('11:00:00', '23:59:59'),
        'Sunday': ('12:00:00', '20:00:00'),
        'Monday': ('12:00:00', '22:00:00'),
        'Tuesday': ('12:00:00', '22:00:00'),
        'Wednesday': ('12:00:00', '23:00:00'),
        'Thursday': ('12:00:00', '23:00:00'),
        'Friday': ('12:00:00', '23:59:59')
    }

    df['Is_Open'] = [
        1 if working_hours[row.name.day_name()][0] <= row.name.time().strftime('%H:%M:%S') <= working_hours[row.name.day_name()][1] else 0
        for _, row in df.iterrows()
    ]

    # Split data into train, validation, and test sets
    split1, split2 = 1272, 1608
    train, val, test = df.iloc[:split1], df.iloc[split1:split2], df.iloc[split2:]

    # Build preprocessing pipeline
    datetime_transformer = DatetimeFeatures(
        variables="index",
        features_to_extract=["hour", "day_of_week", "weekend", "month", 'day_of_month']
    )
    cyclical = CyclicalFeatures(variables=['hour', 'day_of_week'])
    lag_transformer = LagFeatures(variables=[target], periods=[168, 192, 216, 264, 240, 288, 336, 312, 215, 263])
    rolling_window_transformer = RollingWindowFeatures(window=24, shift_period=169)

    pipe = Pipeline([
        ("datetime", datetime_transformer),
        ("cyclical", cyclical),
        ("lag", lag_transformer),
        ("rolling_window", rolling_window_transformer)
    ])

    # Apply transformations and fill NaN values with 0
    train = pipe.fit_transform(train).fillna(0)
    val = pipe.transform(val).fillna(0)
    test = pipe.transform(test).fillna(0)

    # Split features and target variables
    X_train, y_train = train.drop([target], axis=1), train[[target]]
    X_val, y_val = val.drop([target], axis=1), val[[target]]
    X_test, y_test = test.drop([target], axis=1), test[[target]]

    # Normalise features
    scaler_X = MinMaxScaler().fit(X_train)
    X_train_normalized = scaler_X.transform(X_train)
    X_val_normalized = scaler_X.transform(X_val)
    X_test_normalized = scaler_X.transform(X_test)

    # Normalise target
    scaler_y = MinMaxScaler().fit(y_train)
    y_train_normalized = scaler_y.transform(y_train)
    y_val_normalized = scaler_y.transform(y_val)
    y_test_normalized = scaler_y.transform(y_test)
    
    # Save target scaler
    joblib.dump(scaler_y, 'scaler_y_elec.joblib')

    return X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, X_test_normalized, y_test_normalized
