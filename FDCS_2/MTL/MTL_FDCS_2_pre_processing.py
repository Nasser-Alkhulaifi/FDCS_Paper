from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import CyclicalFeatures
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import joblib

# Now target is a list of columns
targets = ['Electricity Consumption', 'Humidity Closer to Evaporator (%)', 'Temperature Closer to Evaporator (C)']

# Custom transformer for calculating rolling window statistics for multiple targets
class RollingWindowFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, targets, window=24, shift_period=169):
        self.window = window
        self.shift_period = shift_period
        self.targets = targets
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        for target in self.targets:
            rolling_windows = X[target].rolling(window=self.window)
            
            rolling_features = pd.DataFrame({
                f'{target}_Mean_192': rolling_windows.mean().shift(self.shift_period),
                f'{target}_Std_192': rolling_windows.std().shift(self.shift_period),
                f'{target}_Max_192': rolling_windows.max().shift(self.shift_period),
                f'{target}_Min_192': rolling_windows.min().shift(self.shift_period),
                f'{target}_Kurt_192': rolling_windows.kurt().shift(self.shift_period),
                f'{target}_Skew_192': rolling_windows.skew().shift(self.shift_period)
            })
            
            X = pd.concat([X, rolling_features], axis=1)
        return X.fillna(0)

# Pipeline for preprocessing data
def preprocess_data(df):
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

    split1 = 1800
    split2 = 2136

    train = df.iloc[:split1, :]
    val = df.iloc[split1:split2, :]
    test = df.iloc[split2:, :]

    datetime_transformer = DatetimeFeatures(
        variables="index",
        features_to_extract=["hour", "day_of_week", "weekend", "month", 'day_of_month'],
    )

    cyclical = CyclicalFeatures(
        variables=['hour', 'day_of_week'],
    )

    lag_transformer = LagFeatures(
        variables=targets,
        periods=[168,169,261,264,335,243,242,336,241,186,263,178,185,184,218,187,188],
    )

    rolling_window_transformer = RollingWindowFeatures(targets=targets, window=24, shift_period=169)

    pipe = Pipeline(
        [
            ("datetime", datetime_transformer),
            ("cyclical", cyclical),
            ("lag", lag_transformer),
            ("rolling_window", rolling_window_transformer)
        ]
    )

    train = pipe.fit_transform(train)
    val = pipe.transform(val)
    test = pipe.transform(test)

    # Fill NaN values with 0
    train = train.fillna(0)
    val = val.fillna(0)
    test = test.fillna(0)
    


    # Split features and target variables
    X_train = train.drop(targets, axis=1)
    y_train = train[targets]

    X_val = val.drop(targets, axis=1)
    y_val = val[targets]

    X_test = test.drop(targets, axis=1)
    y_test = test[targets]

    # For features
    scaler_X = MinMaxScaler().fit(X_train)
    X_train_normalized = scaler_X.transform(X_train)
    X_val_normalized = scaler_X.transform(X_val)
    X_test_normalized = scaler_X.transform(X_test)


    # For targets - use a separate scaler for each target due to potentially different scales
    scalers_y = {target: MinMaxScaler().fit(y_train[[target]]) for target in targets}
    y_train_normalized = np.hstack([scalers_y[target].transform(y_train[[target]]) for target in targets])
    y_val_normalized = np.hstack([scalers_y[target].transform(y_val[[target]]) for target in targets])
    y_test_normalized = np.hstack([scalers_y[target].transform(y_test[[target]]) for target in targets])
    joblib.dump(scalers_y, 'scalers_y_multi_targets.joblib')

    return X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, X_test_normalized, y_test_normalized

