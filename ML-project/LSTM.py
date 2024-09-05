import pandas as pd
import numpy as np
from keras.losses import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna


# Add RUL column function
def add_rul_column(df):
    max_cycle_per_id = df.groupby('id')['cycle'].max().reset_index()
    max_cycle_per_id.columns = ['id', 'max_cycle']
    df = pd.merge(df, max_cycle_per_id, on='id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop(columns=['max_cycle'])
    return df


# Load the provided CSV files
train_FD001_path = 'train_FD001.csv'
test_FD001_path = 'test_FD001.csv'
RUL_FD001_path = 'RUL_FD001.csv'

# Load data into pandas DataFrames
df_train_FD001 = pd.read_csv(train_FD001_path, header=None)
df_test_FD001 = pd.read_csv(test_FD001_path, header=None)
df_RUL_FD001 = pd.read_csv(RUL_FD001_path, header=None)

# Set column names
column_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
df_train_FD001.columns = column_names
df_test_FD001.columns = column_names

# Add RUL column
df_train_FD001 = add_rul_column(df_train_FD001)
df_RUL_FD001.columns = ['RUL']
df_test_FD001 = df_test_FD001.groupby('id').last().reset_index()
df_test_FD001['RUL'] = df_RUL_FD001['RUL']

# Remove unnecessary columns
pre_df_train_FD001 = df_train_FD001.drop(columns=['id', 'setting1', 'setting2', 'setting3'])
pre_df_test_FD001 = df_test_FD001.drop(columns=['id', 'setting1', 'setting2', 'setting3'])

# Define features and target
X_train = pre_df_train_FD001.drop(columns=['RUL'])
y_train = pre_df_train_FD001['RUL']
X_test = pre_df_test_FD001.drop(columns=['RUL'])
y_test = pre_df_test_FD001['RUL']
y_test = y_test.clip(upper=125)
y_train = y_train.clip(upper=125)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled data back to DataFrame
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Reshape input to be [samples, time steps, features]
X_train_reshaped = X_train_scaled_df.values.reshape((X_train_scaled_df.shape[0], 1, X_train_scaled_df.shape[1]))
X_test_reshaped = X_test_scaled_df.values.reshape((X_test_scaled_df.shape[0], 1, X_test_scaled_df.shape[1]))


def create_model(trial):
    # Suggest hyperparameters for the model
    lstm_units1 = trial.suggest_int('lstm_units1', 50, 200)
    lstm_units2 = trial.suggest_int('lstm_units2', 50, 200)
    lstm_units3 = trial.suggest_int('lstm_units3', 50, 200)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

    model = Sequential()
    model.add(Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    model.add(LSTM(lstm_units1, activation='relu', return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units2, activation='relu', return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units3, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    return model


def objective(trial):
    model = create_model(trial)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train_reshaped, y_train, epochs=100, batch_size=64, validation_data=(X_test_reshaped, y_test),
                        verbose=0, shuffle=False, callbacks=[early_stopping])

    y_pred = model.predict(X_test_reshaped)
    mse = mean_squared_error(y_test, y_pred)

    return mse


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Train the final model with the best hyperparameters
best_trial = study.best_trial
model = create_model(best_trial)
history = model.fit(X_train_reshaped, y_train, epochs=100, batch_size=64, validation_data=(X_test_reshaped, y_test),
                    verbose=2, shuffle=False,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])

# Evaluate the model
y_pred = model.predict(X_test_reshaped)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'Mean Absolute Error: {mse}')
print(f'R2 Score: {r2}')
print(f'Mean Absolute Percentage Error: {mape}%')

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plot predicted vs actual RUL
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual RUL')
plt.plot(y_pred, label='Predicted RUL')
plt.title('Actual vs Predicted RUL')
plt.ylabel('RUL')
plt.xlabel('Samples')
plt.legend(loc='upper right')
plt.show()
