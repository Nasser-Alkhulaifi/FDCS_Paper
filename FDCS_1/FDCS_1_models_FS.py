import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, SelectFromModel, RFE
from sklearn.linear_model import Lasso, LinearRegression, ElasticNet
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Feature selection function
def feature_selection(X, y, method):
    return X if method is None else method.fit_transform(X, y)

# Function to train and evaluate multiple models with feature selection methods
def run_model(X_train_val, y_train_val, X_test, y_test, output_names):
    """
    Run multiple regression models with various feature selection methods and evaluate their performance.

    Parameters:
    -----------
    X_train_val : numpy.ndarray
        Training and validation features.
    y_train_val : numpy.ndarray
        Training and validation targets.
    X_test : numpy.ndarray
        Test features.
    y_test : numpy.ndarray
        Test targets.
    output_names : list of str
        Names of the output variables.

    Returns:
    --------
    pred_true_TestSet : pandas.DataFrame
        DataFrame containing true and predicted values for each model and feature selection method.
    """
    start_time = time.time()

    # Define feature selection methods
    feature_selection_methods = {
        "Without": None,
        "Filter - Correlation": SelectKBest(f_regression, k=20),
        "Filter - Mutual Information": SelectKBest(mutual_info_regression, k=20),
        "Embedded - Lasso": SelectFromModel(Lasso(alpha=0.01)),
        "Embedded - Tree Importance (Extra Trees)": SelectFromModel(ExtraTreesRegressor(n_estimators=300)),
        "Embedded - ElasticNet": SelectFromModel(ElasticNet(alpha=0.01, l1_ratio=0.5)),
        "Wrapper - RFE (LinearRegression)": RFE(estimator=LinearRegression(), n_features_to_select=20),
        "Wrapper - Forward Selection": SFS(LinearRegression(), k_features=20, forward=True, floating=False, scoring='neg_mean_squared_error', cv=5),
        "Hybrid - Filter (SelectKBest) + Embedded (RF)": SelectFromModel(RandomForestRegressor(n_estimators=300))
    }

    # Define model parameters
    knn_params = {'leaf_size': 10, 'n_neighbors': 10, 'weights': 'distance'}
    rf_params = {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 500}
    mlp_params = {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 16,
                  'hidden_layer_sizes': (50, 100), 'learning_rate_init': 0.01,
                  'max_iter': 500, 'solver': 'adam'}
    xgb_params = {'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 2,
                  'n_estimators': 500, 'subsample': 0.8, 'learning_rate': 0.01}

    # Define models
    models = {
        'KNeighborsRegressor': KNeighborsRegressor(**knn_params),
        'RandomForestRegressor': RandomForestRegressor(**rf_params),
        'MLPRegressor': MLPRegressor(**mlp_params),
        'XGBRegressor': XGBRegressor(**xgb_params)
    }

    tscv = TimeSeriesSplit(n_splits=5)
    pred_true_TestSet = pd.DataFrame()

    # Iterate through feature selection methods and models
    for fs_method_name, fs_method in feature_selection_methods.items():
        X_train_val_fs = feature_selection(X_train_val, y_train_val, fs_method)
        X_test_fs = X_test if fs_method is None else fs_method.transform(X_test)

        for model_name, model in models.items():
            for i, output_name in enumerate(output_names):
                model_predictions = {'Model': [], f'True_{output_name}': [], f'Pred_{output_name}': []}

                # Cross-validation using TimeSeriesSplit
                for fold, (train_index, val_index) in enumerate(tscv.split(X_train_val_fs)):
                    X_train_fold, X_val_fold = X_train_val_fs[train_index], X_train_val_fs[val_index]
                    y_train_fold, y_val_fold = y_train_val[train_index, i], y_train_val[val_index, i]
                    model.fit(X_train_fold, y_train_fold)

                # Fit model to the entire training set and predict on the test set
                model.fit(X_train_val_fs, y_train_val[:, i])
                test_predictions = model.predict(X_test_fs)

                # Ensure prediction length matches test data
                assert len(test_predictions) == len(y_test[:, i]), "Length of predictions and true values does not match."

                model_predictions['Model'] += [f"{model_name} ({fs_method_name})"] * len(test_predictions)
                model_predictions[f'True_{output_name}'] += list(y_test[:, i])
                model_predictions[f'Pred_{output_name}'] += list(test_predictions)

                # Concatenate results
                temp_df = pd.DataFrame(model_predictions)
                pred_true_TestSet = pd.concat([pred_true_TestSet, temp_df], ignore_index=True)

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    return pred_true_TestSet
