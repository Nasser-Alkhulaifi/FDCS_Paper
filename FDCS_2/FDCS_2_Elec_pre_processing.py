from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, SelectFromModel
from sklearn.linear_model import Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import time

look_back = 24

# Function to create datasets for LSTM model
def create_dataset(X, Y, look_back=look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:(i + look_back)])
        ys.append(Y[i + look_back])
    return np.array(Xs), np.array(ys)

# Function to train and evaluate the LSTM model
def run_lstm_model(X_train_val, y_train_val, X_test, y_test, output_names):
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

    pred_true_TestSet = pd.DataFrame()

    for name, method in feature_selection_methods.items():
        # Apply feature selection method
        if method is not None:
            X_train_val_selected = method.fit_transform(X_train_val, y_train_val)
            X_test_selected = method.transform(X_test)
        else:
            X_train_val_selected, X_test_selected = X_train_val, X_test

        # Reshape datasets for LSTM
        X_train_val_reshaped, y_train_val_reshaped = create_dataset(X_train_val_selected, y_train_val)
        X_test_reshaped, y_test_reshaped = create_dataset(X_test_selected, y_test)

        # Define LSTM model
        model = Sequential([
            LSTM(20, input_shape=(look_back, X_train_val_reshaped.shape[2]), return_sequences=True),
            LSTM(10, return_sequences=False),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
        tscv = TimeSeriesSplit(n_splits=5)

        # Train and evaluate model
        for i, output_name in enumerate(output_names): 
            model_predictions = {'Model': [], 'True': [], 'Pred': []}

            # Time series cross-validation
            for fold, (train_index, val_index) in enumerate(tscv.split(X_train_val_reshaped)):
                X_train_fold, X_val_fold = X_train_val_reshaped[train_index], X_train_val_reshaped[val_index]
                y_train_fold, y_val_fold = y_train_val_reshaped[train_index], y_train_val_reshaped[val_index]
                model.fit(X_train_fold, y_train_fold, epochs=100, batch_size=32, validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping])

            # Make predictions on test set
            test_predictions = model.predict(X_test_reshaped)
            model_predictions['Model'] = [f"LSTM + {name}"] * len(test_predictions)
            model_predictions['True'] = y_test_reshaped.flatten().tolist()
            model_predictions['Pred'] = test_predictions.flatten().tolist()

            # Store predictions in DataFrame
            temp_df = pd.DataFrame(model_predictions)
            pred_true_TestSet = pd.concat([pred_true_TestSet, temp_df], ignore_index=True)

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    return pred_true_TestSet
