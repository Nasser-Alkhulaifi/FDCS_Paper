from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pandas as pd
from feature_engine.timeseries.forecasting import LagFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import CyclicalFeatures

target = 'Humidity Closer to Evaporator (%)'
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
            target + '_Mean_192': rolling_windows.mean().shift(self.shift_period),
            target + '_Std_192': rolling_windows.std().shift(self.shift_period),
            target + '_Max_192': rolling_windows.max().shift(self.shift_period),
            target + '_Min_192': rolling_windows.min().shift(self.shift_period),
            target + '_Kurt_192': rolling_windows.kurt().shift(self.shift_period),
            target + '_Skew_192': rolling_windows.skew().shift(self.shift_period)
        })
        
        return pd.concat([X, rolling_features], axis=1).fillna(0)

# Pipeline for preprocessing data
def preprocess_data(df):
    """
    Preprocesses the input DataFrame by performing the following steps:
    
    1. Converts the 'DateTime' column to datetime format and sets it as the index.
    2. Adds an 'Is_Open' column indicating whether the business is open based on predefined working hours.
    3. Splits the DataFrame into training, validation, and test sets.
    4. Applies a series of transformations including datetime features, cyclical features, lag features, and rolling window features.
    5. Fills NaN values with 0.
    6. Splits the data into features (X) and target (y) variables for training, validation, and test sets.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data to be preprocessed.

    Returns:
        tuple: A tuple containing the following elements:
            - X_train (pd.DataFrame): Features for the training set.
            - y_train (pd.DataFrame): Target variable for the training set.
            - X_val (pd.DataFrame): Features for the validation set.
            - y_val (pd.DataFrame): Target variable for the validation set.
            - X_test (pd.DataFrame): Features for the test set.
            - y_test (pd.DataFrame): Target variable for the test set.
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

    split1 = 1272
    split2 = 1608

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
        variables=[target],
        periods=[168, 192, 169, 191, 216, 170, 240, 264, 193, 171],
    )

    rolling_window_transformer = RollingWindowFeatures(window=24, shift_period=169)

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
    X_train = train.drop([target], axis=1)
    y_train = train[[target]]

    X_val = val.drop([target], axis=1)
    y_val = val[[target]]

    X_test = test.drop([target], axis=1)
    y_test = test[[target]]

    return X_train, y_train, X_val, y_val, X_test, y_test

