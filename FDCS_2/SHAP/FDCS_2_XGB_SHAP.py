# FDCS_1_XGB.py
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import time

def run_xgb_model(X_train_val_normalized, y_train_val_normalized, X_test_normalized, y_test_normalized, output_names):
    """
    Train and evaluate an XGBoost model using time series cross-validation.
    
    Parameters:
    - X_train_val_normalized: Normalized training + validation feature data.
    - y_train_val_normalized: Normalized training + validation target data.
    - X_test_normalized: Normalized test feature data.
    - y_test_normalized: Normalized test target data.
    - output_names: Names of the target variables.
    """
    start_time = time.time()

    best_params = {
        'max_depth': 30,
        'learning_rate': 0.01,
        'min_child_weight': 30,
        'n_estimators': 500,
        'gamma': 0,
        'subsample': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.5,
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'random_state': 42
    }

    tscv = TimeSeriesSplit(n_splits=5)

    metrics_df = pd.DataFrame(columns=['Fold', 'Set', 'Model', 'Output', 'MAE', 'RMSE'])

    for i in range(y_train_val_normalized.shape[1]):
        val_predictions = np.zeros(y_train_val_normalized.shape[0])
        val_indices = []

        for fold, (train_index, val_index) in enumerate(tscv.split(X_train_val_normalized)):
            X_train_fold, X_val_fold = X_train_val_normalized[train_index], X_train_val_normalized[val_index]
            y_train_fold, y_val_fold = y_train_val_normalized[train_index, i], y_train_val_normalized[val_index, i]

            model = XGBRegressor(**best_params)
            model.fit(X_train_fold, y_train_fold)

            predictions = model.predict(X_val_fold)
            val_predictions[val_index] = predictions
            val_indices.extend(val_index)

            mae = mean_absolute_error(y_val_fold, predictions)
            mse = mean_squared_error(y_val_fold, predictions)
            rmse = sqrt(mse)
            metrics_df = metrics_df.append({'Fold': fold+1, 'Set': 'Cross-Validation', 'Model': 'XGBoost', 'Output': output_names[i], 'MAE': mae, 'RMSE': rmse}, ignore_index=True)

        model.fit(X_train_val_normalized, y_train_val_normalized[:, i])
        test_predictions = model.predict(X_test_normalized)

        mae = mean_absolute_error(y_test_normalized[:, i], test_predictions)
        mse = mean_squared_error(y_test_normalized[:, i], test_predictions)
        rmse = sqrt(mse)

        metrics_df = metrics_df.append({'Fold': 'Final', 'Set': 'Test', 'Model': 'XGBoost', 'Output': output_names[i], 'MAE': mae, 'RMSE': rmse}, ignore_index=True)

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    return metrics_df.round(2)
