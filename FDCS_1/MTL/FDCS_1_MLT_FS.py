import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from tensorflow.keras.callbacks import EarlyStopping

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

def evaluate_model(X_train_val, y_train_val, X_test, y_test, output_names):
    """
    Evaluates a multi-task learning model using different feature selection methods and time series cross-validation.

    Parameters:
    X_train_val (numpy.ndarray): Training and validation features.
    y_train_val (numpy.ndarray): Training and validation targets.
    X_test (numpy.ndarray): Test features.
    y_test (numpy.ndarray): Test targets.
    output_names (list): List of output names corresponding to the tasks.

    Returns:
    pandas.DataFrame: DataFrame containing true and predicted values for the test set, along with model, output, and feature selection method information.
    """
    results = []
    tscv = TimeSeriesSplit(n_splits=5)
    num_tasks = len(output_names)

    for name, selector in feature_selection_methods.items():
        if selector is not None:
            selector.fit(X_train_val, y_train_val[:, 0])  # Assuming y_train_val[:, 0] for fitting, adjust as necessary
            X_train_val_selected = selector.transform(X_train_val)
            X_test_selected = selector.transform(X_test)
        else:
            X_train_val_selected = X_train_val
            X_test_selected = X_test

        base_model = keras.Sequential([
            layers.Dense(10, activation='relu', input_shape=(X_train_val_selected.shape[1],)),
            layers.Dense(40, activation='relu')
        ])
        inputs = keras.Input(shape=(X_train_val_selected.shape[1],))
        base_output = base_model(inputs)
        outputs = [layers.Dense(1)(base_output) for _ in range(num_tasks)]
        model = keras.Model(inputs=inputs, outputs=outputs)

        optimizer = keras.optimizers.Adam(learning_rate=0.005)
        model.compile(optimizer=optimizer, loss=['mse']*num_tasks, metrics=['mse'], loss_weights=[1]*num_tasks)
        early_stopping = EarlyStopping(monitor='loss', patience=50, verbose=1)

        for fold, (train_index, val_index) in enumerate(tscv.split(X_train_val_selected)):
            X_train, X_val = X_train_val_selected[train_index], X_train_val_selected[val_index]
            y_train, y_val = [y_train_val[train_index, i] for i in range(num_tasks)], [y_train_val[val_index, i] for i in range(num_tasks)]
            model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])

        test_predictions = model.predict(X_test_selected)
        pred_true_TestSet = pd.DataFrame()

        for i, output_name in enumerate(output_names):
            pred_true_df = pd.DataFrame({'True': y_test[:, i], 'Pred': np.squeeze(test_predictions[i])})
            pred_true_df['Model'] = 'MTL'
            pred_true_df['Output'] = output_name
            pred_true_df['Feature_Selection'] = name
            pred_true_TestSet = pd.concat([pred_true_TestSet, pred_true_df], ignore_index=True)

        results.append(pred_true_TestSet)

    final_df = pd.concat(results, ignore_index=True)

    return final_df
