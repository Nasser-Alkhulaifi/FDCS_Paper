import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

def run_model(X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, output_names):
    start_time = time.time()

    tscv = TimeSeriesSplit(n_splits=5)

    metrics_df = pd.DataFrame(columns=['Model', 'Output', 'Best Params', 'MAE', 'RMSE'])
    best_params_df = pd.DataFrame(columns=['Model', 'Output', 'Best Params'])
    pred_true_ValSet = pd.DataFrame()

    models = {
        'KNeighborsRegressor': KNeighborsRegressor(),
        'RandomForestRegressor': RandomForestRegressor(),
        'MLPRegressor': MLPRegressor(),
        'XGBRegressor': XGBRegressor()  
    }
    
    param_grid = {
    'KNeighborsRegressor': {'n_neighbors': [10, 20],
                            'weights': ['uniform','distance'],
                            'leaf_size': [10, 20]},

    'RandomForestRegressor': {'n_estimators': [100, 500],
                              'max_depth': [10, 20],
                              'min_samples_split': [2,10],
                              'min_samples_leaf': [2, 10]},

    'MLPRegressor': {'hidden_layer_sizes': [(100,), (50, 100)],
                     'activation': ['tanh', 'relu'],
                     'solver': ['sgd', 'adam'],
                     'alpha': [0.0001, 0.001],
                     'learning_rate_init': [0.001, 0.01],
                     'max_iter': [100, 500],
                     'batch_size': [16,32]},

    'XGBRegressor': {'n_estimators': [100, 500],
                     'max_depth': [10, 20],
                     'learning_rate': [0.01, 0.1],
                     'min_child_weight': [2, 10],
                     'subsample': [0.8, 1],
                     'learning_rate': [0.3, 0.01]}
}

    for model_name, model in models.items():
        for i, output_name in enumerate(output_names):
            grid_search = GridSearchCV(model, param_grid[model_name], cv=tscv, scoring='neg_mean_absolute_error')
            grid_search.fit(X_train_normalized, y_train_normalized[:, i])
            best_model = grid_search.best_estimator_
            
            predictions = best_model.predict(X_val_normalized)
            mae = mean_absolute_error(y_val_normalized[:, i], predictions)
            mse = mean_squared_error(y_val_normalized[:, i], predictions)
            rmse = sqrt(mse)
            
            metrics_row = pd.DataFrame([{'Model': model_name, 'Output': output_name, 'Best Params': str(grid_search.best_params_), 'MAE': mae, 'RMSE': rmse}])
            metrics_df = pd.concat([metrics_df, metrics_row], ignore_index=True)
            
            best_params_row = pd.DataFrame([{'Model': model_name, 'Output': output_name, 'Best Params': str(grid_search.best_params_)}])
            best_params_df = pd.concat([best_params_df, best_params_row], ignore_index=True)
            
            # Construct prediction DataFrame for current model and output
            model_predictions_df = pd.DataFrame({
                'Model': [model_name] * len(predictions),
                'Best Params': [str(grid_search.best_params_)] * len(predictions),
                f'True_{output_name}': y_val_normalized[:, i],
                f'Pred_{output_name}': predictions
            })
            
            # This is the line you were missing
            pred_true_ValSet = pd.concat([pred_true_ValSet, model_predictions_df], ignore_index=True)

    
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    print("Best Parameters for Each Model and Output:")
    print(best_params_df.to_string(index=False))
    return metrics_df.round(2), pred_true_ValSet, best_params_df
