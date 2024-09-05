import warnings
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true.reset_index(drop=True), label='True Values')
    plt.plot(y_pred, label='Predicted Values', linestyle='--')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('RUL')
    plt.legend()
    plt.show()

# RUL 컬럼을 추가하는 함수 정의
def add_rul_column(df):
    max_cycle_per_id = df.groupby('id')['cycle'].max().reset_index()
    max_cycle_per_id.columns = ['id', 'max_cycle']
    df = pd.merge(df, max_cycle_per_id, on='id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop(columns=['max_cycle'])
    return df

# 데이터 로드
df_train_FD001 = pd.read_csv('train_FD001.csv', header=None)
df_test_FD001 = pd.read_csv('test_FD001.csv', header=None)
df_RUL_FD001 = pd.read_csv('RUL_FD001.csv', header=None)

# 컬럼명 설정
column_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
df_train_FD001.columns = column_names
df_test_FD001.columns = column_names

# RUL 컬럼 추가
df_train_FD001 = add_rul_column(df_train_FD001)
df_RUL_FD001.columns = ['RUL']
df_test_FD001 = df_test_FD001.groupby('id').last().reset_index()
df_test_FD001['RUL'] = df_RUL_FD001['RUL']

# 불필요한 컬럼 제거
pre_df_train_FD001 = df_train_FD001.drop(
    columns=['id', 'setting1', 'setting2', 'setting3', 'sensor1', 'sensor5', 'sensor6', 'sensor9', 'sensor10',
             'sensor14', 'sensor16', 'sensor18', 'sensor19'])
pre_df_test_FD001 = df_test_FD001.drop(
    columns=['id', 'setting1', 'setting2', 'setting3', 'sensor1', 'sensor5', 'sensor6', 'sensor9', 'sensor10',
             'sensor14', 'sensor16', 'sensor18', 'sensor19'])

# 특징(feature)와 목표(target) 설정
X_train = pre_df_train_FD001.drop(columns=['RUL'])
y_train = pre_df_train_FD001['RUL']
X_test = pre_df_test_FD001.drop(columns=['RUL'])
y_test = pre_df_test_FD001['RUL']
y_test= y_test.clip(upper=125)
y_train= y_train.clip(upper=125)

# 데이터 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Optuna를 이용한 하이퍼파라미터 튜닝
def objective(trial):
    param = {
        'C': trial.suggest_loguniform('C', 1e-3, 1e2),
        'epsilon': trial.suggest_loguniform('epsilon', 1e-4, 1e-1),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'degree': trial.suggest_int('degree', 2, 5) if trial.params['kernel'] == 'poly' else 3
    }

    model = SVR(**param)
    model.fit(X_train_scaled_df, y_train)

    preds = model.predict(X_test_scaled_df)
    mse = mean_squared_error(y_test, preds)
    return mse

# 하이퍼파라미터 튜닝 수행
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)
best_params = study.best_params

# 최적의 하이퍼파라미터로 모델 학습
model = SVR(**best_params)
model.fit(X_train_scaled_df, y_train)

# 예측 및 평가
y_pred_train = model.predict(X_train_scaled_df)
y_pred_test = model.predict(X_test_scaled_df)

mse_train = mean_squared_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Train MSE: {mse_train}, Train R^2: {r2_train}")
print(f"Test MSE: {mse_test}, Test R^2: {r2_test}")

# y_test['RUL']값이 70을 기준으로 테스트 세트를 두 개로 나누기
mask = y_test > 70
X_test_high = X_test_scaled_df[mask]
y_test_high = y_test[mask]
X_test_low = X_test_scaled_df[~mask]
y_test_low = y_test[~mask]

# 두 개의 테스트 세트에 대해 예측 수행 및 평가
y_pred_test_high = model.predict(X_test_high)
y_pred_test_low = model.predict(X_test_low)

mse_test_high = mean_squared_error(y_test_high, y_pred_test_high)
r2_test_high = r2_score(y_test_high, y_pred_test_high)
mse_test_low = mean_squared_error(y_test_low, y_pred_test_low)
r2_test_low = r2_score(y_test_low, y_pred_test_low)

print(f"Test High RUL MSE: {mse_test_high}, Test High R^2: {r2_test_high}")
print(f"Test Low RUL MSE: {mse_test_low}, Test Low R^2: {r2_test_low}")

# 두 개의 테스트 세트에 대한 예측값 시각화
plot_predictions(y_test_high, y_pred_test_high, 'SVM Predictions vs True Values (High RUL)')
plot_predictions(y_test_low, y_pred_test_low, 'SVM Predictions vs True Values (Low RUL)')
