import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def run_model(X_train, y_train, X_test, y_test, output_names):
    start_time = time.time()

    rf_params = {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 500}
    xgb_params = {'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 2, 'n_estimators': 500, 'subsample': 0.8}

    dataset_sizes = list(range(24, X_train.shape[0]+1, 24))  # Assuming stepping by 24
    models = {
        'RandomForestRegressor': RandomForestRegressor(**rf_params),
        'XGBRegressor': XGBRegressor(**xgb_params)
    }

    pred_true_TestSet = pd.DataFrame()

    for model_name, model in models.items():
        for size in dataset_sizes:
            X_train_subset = X_train[:size]
            y_train_subset = y_train[:size]
            model.fit(X_train_subset, y_train_subset)

            for i, output_name in enumerate(output_names):
                test_predictions = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test[:, i], test_predictions))

                model_results = {
                    'Model': f"{model_name}",
                    'TrainSize': size,
                    'RMSE': rmse,
                    f'True_{output_name}': y_test[:, i],
                    f'Pred_{output_name}': test_predictions
                }

                if pred_true_TestSet.empty:
                    pred_true_TestSet = pd.DataFrame(model_results)
                else:
                    temp_df = pd.DataFrame(model_results)
                    pred_true_TestSet = pd.concat([pred_true_TestSet, temp_df], ignore_index=True)

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    return pred_true_TestSet

