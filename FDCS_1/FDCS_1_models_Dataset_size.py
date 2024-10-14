import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def run_model(X_train, y_train, X_test, y_test, output_names):
    """
    Trains and evaluates multiple regression models on varying sizes of the training dataset.

    Parameters:
    X_train (pd.DataFrame or np.ndarray): Training feature set.
    y_train (pd.DataFrame or np.ndarray): Training target set.
    X_test (pd.DataFrame or np.ndarray): Testing feature set.
    y_test (pd.DataFrame or np.ndarray): Testing target set.
    output_names (list of str): List of output names corresponding to the columns in y_test.

    Returns:
    pd.DataFrame: A DataFrame containing the model name, training size, RMSE, true values, and predicted values for each output.
    """
    start_time = time.time()

    # Define model parameters
    rf_params = {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 500}
    xgb_params = {'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 2, 'n_estimators': 500, 'subsample': 0.8}

    # Define models
    models = {
        'RandomForestRegressor': RandomForestRegressor(**rf_params),
        'XGBRegressor': XGBRegressor(**xgb_params)
    }

    dataset_sizes = range(24, X_train.shape[0] + 1, 24)  # Assuming step size of 24
    pred_true_TestSet = pd.DataFrame()

    # Iterate over models and dataset sizes
    for model_name, model in models.items():
        for size in dataset_sizes:
            X_train_subset = X_train[:size]
            y_train_subset = y_train[:size]

            # Train the model on the current subset
            model.fit(X_train_subset, y_train_subset)

            # Make predictions for each output
            for i, output_name in enumerate(output_names):
                test_predictions = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test[:, i], test_predictions))

                model_results = {
                    'Model': model_name,
                    'TrainSize': size,
                    'RMSE': rmse,
                    f'True_{output_name}': y_test[:, i],
                    f'Pred_{output_name}': test_predictions
                }

                temp_df = pd.DataFrame([model_results])  # Create a DataFrame for the current results
                pred_true_TestSet = pd.concat([pred_true_TestSet, temp_df], ignore_index=True)

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    return pred_true_TestSet
