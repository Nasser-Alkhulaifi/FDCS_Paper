import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

def evaluate_model(X_train_val_normalized, y_train_val_normalized, X_test_normalized, y_test_normalized, output_names):
    tscv = TimeSeriesSplit(n_splits=5)
    num_tasks = len(output_names)

    def create_base_model(input_shape):
        model = keras.Sequential([
            layers.Dense(10, activation='relu', input_shape=(input_shape,)),
            layers.Dense(40, activation='relu')
        ])
        return model

    def create_multi_task_model(base_model, num_tasks):
        inputs = keras.Input(shape=(base_model.input_shape[1],))
        base_output = base_model(inputs)
        outputs = [layers.Dense(1)(base_output) for _ in range(num_tasks)]
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model

    base_model = create_base_model(X_train_val_normalized.shape[1])
    model = create_multi_task_model(base_model, num_tasks=num_tasks)

    optimizer = keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer=optimizer, loss=['mse']*num_tasks, metrics=['mse'], loss_weights=[1]*num_tasks)
    early_stopping = EarlyStopping(monitor='loss', patience=50, verbose=1)

    metrics_df = pd.DataFrame(columns=['Fold', 'Set', 'Model', 'Output', 'MAE', 'RMSE'])
    pred_true_TestSet = pd.DataFrame()

    for fold, (train_index, val_index) in enumerate(tscv.split(X_train_val_normalized)):
        X_train, X_val = X_train_val_normalized[train_index], X_train_val_normalized[val_index]
        y_train, y_val = [y_train_val_normalized[train_index, i] for i in range(num_tasks)], [y_train_val_normalized[val_index, i] for i in range(num_tasks)]
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])
        
        val_predictions = model.predict(X_val)
        for i, output_name in enumerate(output_names):
            mae = mean_absolute_error(y_val[i], np.squeeze(val_predictions[i]))
            mse = mean_squared_error(y_val[i], np.squeeze(val_predictions[i]))
            rmse = sqrt(mse)
            new_row = pd.DataFrame({'Fold': [fold + 1], 'Set': ['Cross-Validation'], 'Model': ['MTL'], 'Output': [output_name], 'MAE': [mae], 'RMSE': [rmse]})
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)

    test_predictions = model.predict(X_test_normalized)
    for i, output_name in enumerate(output_names):
        mae = mean_absolute_error(y_test_normalized[:, i], np.squeeze(test_predictions[i]))
        mse = mean_squared_error(y_test_normalized[:, i], np.squeeze(test_predictions[i]))
        rmse = sqrt(mse)
        new_test_row = pd.DataFrame({'Fold': ['Final'], 'Set': ['Test'], 'Model': ['MTL'], 'Output': [output_name], 'MAE': [mae], 'RMSE': [rmse]})
        metrics_df = pd.concat([metrics_df, new_test_row], ignore_index=True)
        
        pred_true_df = pd.DataFrame({'True': y_test_normalized[:, i], 'Pred': np.squeeze(test_predictions[i])})
        pred_true_df['Model'] = 'MTL'
        pred_true_df['Output'] = output_name
        pred_true_TestSet = pd.concat([pred_true_TestSet, pred_true_df], ignore_index=True)

    return metrics_df, pred_true_TestSet
