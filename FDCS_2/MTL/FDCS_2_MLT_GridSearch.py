import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner import HyperModel, RandomSearch

class MultiTaskHyperModel(HyperModel):
    def __init__(self, input_shape, num_tasks):
        self.input_shape = input_shape
        self.num_tasks = num_tasks

    def build(self, hp):
        base_model = keras.Sequential([
            layers.Dense(
                units=hp.Int('units_1', min_value=10, max_value=40, step=10),
                activation='relu',
                input_shape=(self.input_shape,)
            ),
            layers.Dense(
                units=hp.Int('units_2', min_value=10, max_value=40, step=10),
                activation='relu'
            )
        ])
        inputs = keras.Input(shape=(base_model.input_shape[1],))
        base_output = base_model(inputs)
        outputs = [layers.Dense(1)(base_output) for _ in range(self.num_tasks)]
        model = keras.Model(inputs=inputs, outputs=outputs)

        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=['mse'] * self.num_tasks,
            metrics=['mse'],
            loss_weights=[1] * self.num_tasks
        )

        return model

def evaluate_model_with_tuner(X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized, output_names):
    input_shape = X_train_normalized.shape[1]
    num_tasks = len(output_names)
    hypermodel = MultiTaskHyperModel(input_shape=input_shape, num_tasks=num_tasks)

    tuner = RandomSearch(
        hypermodel,
        objective='val_loss',
        max_trials=10,  # Adjust based on computational budget
        executions_per_trial=2,
        directory='my_dir',
        project_name='multi_task_learning',
        overwrite=True,
    )

    tscv = TimeSeriesSplit(n_splits=5)
    metrics_df = pd.DataFrame(columns=['Fold', 'Set', 'Model', 'Output', 'MAE', 'RMSE'])
    pred_true_ValSet = pd.DataFrame()

    for fold, (train_index, val_index) in enumerate(tscv.split(X_train_normalized)):
        X_train, X_val = X_train_normalized[train_index], X_train_normalized[val_index]
        y_train, y_val = [y_train_normalized[train_index, i] for i in range(num_tasks)], [y_train_normalized[val_index, i] for i in range(num_tasks)]
        
        # Note: Adjust 'epochs' and 'validation_data'
        tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    # Now retrain the model with the best hyperparameters on the full training set
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    model.fit(X_train_normalized, y_train_normalized, epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])

    # Evaluation on the test set
    test_predictions = model.predict(X_val_normalized)
    for i, output_name in enumerate(output_names):
        mae = mean_absolute_error(y_val_normalized[:, i], np.squeeze(test_predictions[i]))
        mse = mean_squared_error(y_val_normalized[:, i], np.squeeze(test_predictions[i]))
        rmse = sqrt(mse)
        new_test_row = pd.DataFrame({
            'Fold': ['Final'], 'Set': ['Test'], 'Model': ['MTL'], 'Output': [output_name], 'MAE': [mae], 'RMSE': [rmse]
        })
        metrics_df = pd.concat([metrics_df, new_test_row], ignore_index=True)
        
        # Prepare DataFrame for true vs predicted values
        pred_true_df = pd.DataFrame({
            'True': y_val_normalized[:, i], 
            'Pred': np.squeeze(test_predictions[i])
        })
        pred_true_df['Model'] = 'MTL'
        pred_true_df['Output'] = output_name
        pred_true_ValSet = pd.concat([pred_true_ValSet, pred_true_df], ignore_index=True)

        best_hyperparams_dict = {
        'units_1': best_hps.get('units_1'),
        'units_2': best_hps.get('units_2'),
        'learning_rate': best_hps.get('learning_rate')
    }

    # Return the best hyperparameters, metrics dataframe, and predictions for the test set
    return best_hyperparams_dict, metrics_df, pred_true_ValSet

