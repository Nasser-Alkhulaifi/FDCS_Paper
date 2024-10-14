import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def run_model(X_train, y_train, X_test, y_test, output_names):
    start_time = time.time()

    # Model parameters
    rf_params = {'max_depth': 10, 'min_samples_leaf': 10, 'min_samples_split': 2, 'n_estimators': 500}
    xgb_params = {'learning_rate': 0.01, 'max_depth': 20, 'min_child_weight': 10, 'n_estimators': 500, 'subsample': 0.8}

    # Define dataset sizes (increment by 24 each time)
    dataset_sizes = list(range(24, X_train.shape[0] + 24, 24))
    
    # Models
    models = {
        'RandomForestRegressor': RandomForestRegressor(**rf_params),
        'XGBRegressor': XGBRegressor(**xgb_params)
    }

    pred_true_TestSet = pd.DataFrame()

    # Loop over models and dataset sizes
    for model_name, model in models.items():
        for size in dataset_sizes:
            X_train_subset = X_train[:size]
            y_train_subset = y_train[:size]
            model.fit(X_train_subset, y_train_subset)

            # Evaluate the model on test data
            for i, output_name in enumerate(output_names):
                test_predictions = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test[:, i], test_predictions))

                # Store the results
                model_results = {
                    'Model': model_name,
                    'TrainSize': size,
                    'RMSE': rmse,
                    f'True_{output_name}': y_test[:, i],
                    f'Pred_{output_name}': test_predictions
                }

                temp_df = pd.DataFrame([model_results])
                pred_true_TestSet = pd.concat([pred_true_TestSet, temp_df], ignore_index=True)

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    return pred_true_TestSet
