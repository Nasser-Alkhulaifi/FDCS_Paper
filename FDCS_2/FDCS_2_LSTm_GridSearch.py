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

# Create dataset for LSTM model
def create_dataset(X, Y, look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:(i + look_back)])
        ys.append(Y[i + look_back])
    return np.array(Xs), np.array(ys)

# Define HyperModel for LSTM
class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape, output_units):
        self.input_shape = input_shape
        self.output_units = output_units

    def build(self, hp):
        model = Sequential()
        model.add(LSTM(units=hp.Choice('units_first_layer', [5, 10, 20, 40]),
                       input_shape=self.input_shape,
                       return_sequences=True))
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(LSTM(units=hp.Choice(f'units_layer_{i+1}', [5, 10, 20, 40]),
                           return_sequences=i < hp.get('num_layers') - 1))
            if hp.Boolean(f'relu_activation_{i}'):
                model.add(Dense(units=hp.Choice(f'dense_units_{i}', [1]), activation='relu'))
        model.add(Dense(self.output_units))
        model.compile(optimizer=Adam(hp.Float('learning_rate', min_value=0.001, max_value=0.1, step=0.01)),
                      loss='mean_squared_error')
        return model

# Train and evaluate the LSTM model using hyperparameter tuning
def run_lstm_model(X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, output_names, look_back=look_back):
    start_time = time.time()
    X_train_val_reshaped, y_train_val_reshaped = create_dataset(X_train_normalized, y_train_normalized, look_back)
    X_test_reshaped, y_test_reshaped = create_dataset(X_val_normalized, y_val_normalized, look_back)

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

    # Loop through each output and fold for cross-validation
    for i, output_name in enumerate(output_names):
        model_predictions = {'Model': [], 'True': [], 'Pred': []}
        
        for fold, (train_index, val_index) in enumerate(tscv.split(X_train_val_reshaped)):
            X_train_fold, X_val_fold = X_train_val_reshaped[train_index], X_train_val_reshaped[val_index]
            y_train_fold, y_val_fold = y_train_val_reshaped[train_index], y_train_val_reshaped[val_index]
            
            tuner.search(X_train_fold, y_train_fold, epochs=100, validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping])
        
        best_model = tuner.get_best_models(num_models=1)[0]

        # Evaluate on validation set
        for fold, (train_index, val_index) in enumerate(tscv.split(X_train_val_reshaped)):
            X_val_fold, y_val_fold = X_train_val_reshaped[val_index], y_train_val_reshaped[val_index]
            predictions = best_model.predict(X_val_fold)
            mae = mean_absolute_error(y_val_fold, predictions)
            rmse = sqrt(mean_squared_error(y_val_fold, predictions))
            metrics_df = pd.concat([metrics_df, pd.DataFrame({
                'Fold': [fold+1], 'Set': ['Cross-Validation'], 'Model': ['LSTM'], 'Output': [output_name], 'MAE': [mae], 'RMSE': [rmse]
            })], ignore_index=True)

        # Evaluate on test set
        test_predictions = best_model.predict(X_test_reshaped)
        mae = mean_absolute_error(y_test_reshaped, test_predictions)
        rmse = sqrt(mean_squared_error(y_test_reshaped, test_predictions))
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'Fold': ['Final'], 'Set': ['Test'], 'Model': ['LSTM'], 'Output': [output_name], 'MAE': [mae], 'RMSE': [rmse]
        })], ignore_index=True)

        # Store predictions
        model_predictions['Model'] = ['LSTM'] * len(test_predictions)
        model_predictions['True'] = list(y_test_reshaped.flatten())
        model_predictions['Pred'] = list(test_predictions.flatten())
        pred_true_ValSet = pd.concat([pred_true_ValSet, pd.DataFrame(model_predictions)], ignore_index=True)

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    # Get best hyperparameters
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    num_layers = best_hyperparameters.get('num_layers')
    relevant_hyperparams = ['units_first_layer', 'num_layers', 'learning_rate'] + \
                           [f'units_layer_{i+1}' for i in range(num_layers)] + \
                           [f'relu_activation_{i}' for i in range(num_layers) if best_hyperparameters.get(f'relu_activation_{i}')] + \
                           [f'dense_units_{i}' for i in range(num_layers) if best_hyperparameters.get(f'relu_activation_{i}')]

    best_hyperparams_dict = {param: best_hyperparameters.get(param) for param in relevant_hyperparams}

    return metrics_df, pred_true_ValSet, best_hyperparams_dict
