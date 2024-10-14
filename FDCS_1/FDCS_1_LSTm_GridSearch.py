# FDCS_1_LSTM.py
# No need to manually tune the number of epochs due to the use of an EarlyStopping callback.
# Model is saved at the best epoch evaluated by the validation data.
# Reference: https://keras.io/guides/keras_tuner/getting_started/

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner import HyperModel, RandomSearch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import time
from sklearn.model_selection import TimeSeriesSplit

look_back = 24

# Create dataset for time series prediction
def create_dataset(X, Y, look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:(i + look_back)])
        ys.append(Y[i + look_back])
    return np.array(Xs), np.array(ys)

# Define LSTM model with hyperparameter tuning
class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape, output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Choice('units_first_layer', [5, 10, 20, 40]),
            input_shape=self.input_shape,
            return_sequences=True
        ))
        
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(LSTM(
                units=hp.Choice(f'units_layer_{i+1}', [5, 10, 20, 40]),
                return_sequences=i < hp.get('num_layers') - 1
            ))
            if hp.Boolean(f'relu_activation_{i}'):
                model.add(Dense(units=hp.Choice(f'dense_units_{i}', [1]), activation='relu'))
                
        model.add(Dense(self.output_units))
        model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=0.001, max_value=0.1, step=0.01)),
                      loss='mean_squared_error')

        return model

# Function to train and evaluate LSTM model with hyperparameter tuning and cross-validation
def run_lstm_model(X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, output_names, look_back=look_back):
    """
    Trains and evaluates an LSTM model using hyperparameter search and time series cross-validation.
    Args:
        X_train_normalized (numpy.ndarray): Normalized training feature data.
        y_train_normalized (numpy.ndarray): Normalized training target data.
        X_val_normalized (numpy.ndarray): Normalized validation feature data.
        y_val_normalized (numpy.ndarray): Normalized validation target data.
        output_names (list): List of output names for the model.
        look_back (int, optional): Number of previous time steps to use as input features.
    Returns:
        metrics_df (pandas.DataFrame): DataFrame containing evaluation metrics (MAE, RMSE) for each fold and the test set.
        pred_true_ValSet (pandas.DataFrame): DataFrame containing true and predicted values for the test set.
        best_hyperparams_dict (dict): Best hyperparameters found during the search.
    """
    start_time = time.time()
    
    # Reshape datasets for time series input
    X_train_val_reshaped, y_train_val_reshaped = create_dataset(X_train_normalized, y_train_normalized, look_back)
    X_test_reshaped, y_test_reshaped = create_dataset(X_val_normalized, y_val_normalized, look_back)

    # Hyperparameter tuning with RandomSearch
    hypermodel = LSTMHyperModel(input_shape=(look_back, X_train_normalized.shape[1]), output_units=1)
    tuner = RandomSearch(
        hypermodel,
        objective='val_loss',
        max_trials=20,
        executions_per_trial=2,
        directory='lstm_tuning',
        project_name='lstm_optimization',
        overwrite=True
    )
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    tscv = TimeSeriesSplit(n_splits=5)
    
    metrics_df = pd.DataFrame(columns=['Fold', 'Set', 'Model', 'Output', 'MAE', 'RMSE'])
    pred_true_ValSet = pd.DataFrame()

    # Train and validate for each output
    for output_name in output_names: 
        model_predictions = {'Model': [], 'True': [], 'Pred': []}
        
        for fold, (train_index, val_index) in enumerate(tscv.split(X_train_val_reshaped)):
            X_train_fold, X_val_fold = X_train_val_reshaped[train_index], X_train_val_reshaped[val_index]
            y_train_fold, y_val_fold = y_train_val_reshaped[train_index], y_train_val_reshaped[val_index]
            
            tuner.search(X_train_fold, y_train_fold, epochs=100, validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping])
        
        best_model = tuner.get_best_models(num_models=1)[0]
        
        # Cross-validation evaluation
        for fold, (train_index, val_index) in enumerate(tscv.split(X_train_val_reshaped)):
            X_val_fold, y_val_fold = X_train_val_reshaped[val_index], y_train_val_reshaped[val_index]
            predictions = best_model.predict(X_val_fold)
            mae = mean_absolute_error(y_val_fold, predictions)
            mse = mean_squared_error(y_val_fold, predictions)
            rmse = sqrt(mse)
            
            metrics_df = pd.concat([metrics_df, pd.DataFrame({
                'Fold': [fold + 1], 'Set': ['Cross-Validation'], 'Model': ['LSTM'],
                'Output': [output_name], 'MAE': [mae], 'RMSE': [rmse]
            })], ignore_index=True)

        # Final test set evaluation
        test_predictions = best_model.predict(X_test_reshaped)
        mae = mean_absolute_error(y_test_reshaped, test_predictions)
        mse = mean_squared_error(y_test_reshaped, test_predictions)
        rmse = sqrt(mse)
        
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'Fold': ['Final'], 'Set': ['Test'], 'Model': ['LSTM'],
            'Output': [output_name], 'MAE': [mae], 'RMSE': [rmse]
        })], ignore_index=True)

        model_predictions['Model'] = ['LSTM'] * len(test_predictions)
        model_predictions['True'] = y_test_reshaped.flatten().tolist()
        model_predictions['Pred'] = test_predictions.flatten().tolist()
        pred_true_ValSet = pd.concat([pred_true_ValSet, pd.DataFrame(model_predictions)], ignore_index=True)

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    
    # Retrieve the best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    num_layers = best_hyperparameters.get('num_layers')
    
    relevant_hyperparams = ['units_first_layer', 'num_layers', 'learning_rate'] + \
                           [f'units_layer_{i+1}' for i in range(num_layers)] + \
                           [f'relu_activation_{i}' for i in range(num_layers) if best_hyperparameters.get(f'relu_activation_{i}')] + \
                           [f'dense_units_{i}' for i in range(num_layers) if best_hyperparameters.get(f'relu_activation_{i}')]
    
    best_hyperparams_dict = {param: best_hyperparameters.get(param) for param in relevant_hyperparams}

    return metrics_df, pred_true_ValSet, best_hyperparams_dict
