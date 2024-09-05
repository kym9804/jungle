import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# RUL 컬럼을 추가하는 함수 정의
def add_rul_column(df):
    max_cycle_per_id = df.groupby('id')['cycle'].max().reset_index()
    max_cycle_per_id.columns = ['id', 'max_cycle']
    df = pd.merge(df, max_cycle_per_id, on='id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop(columns=['max_cycle'])
    return df

def apply_smoothing(data, window_size=5):
    return data.rolling(window=window_size, min_periods=1).mean()

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
y_test = y_test.clip(upper=125)
y_train = y_train.clip(upper=125)


# 데이터 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(y_train)
X_test_scaled = scaler.transform(y_test)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# PCA 적용
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled_df)
X_test_pca = pca.transform(X_test_scaled_df)

# Optuna를 사용한 하이퍼파라미터 튜닝
def objective(trial):
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train_scaled_df, y_train)
    y_pred = model.predict(X_train_scaled_df)
    mse = mean_squared_error(y_train, y_pred)
    return mse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# 최적의 하이퍼파라미터로 모델 학습
best_params = study.best_params
model = LinearRegression(**best_params)
model.fit(X_train_scaled_df, y_train)

# 예측값 계산
y_train_pred = model.predict(X_train_scaled_df)
y_test_pred = model.predict(X_test_scaled_df)

# MSE와 R^2 계산
mse_train_lr = mean_squared_error(y_train, y_train_pred)
r2_train_lr = r2_score(y_train, y_train_pred)
mse_test_lr = mean_squared_error(y_test, y_test_pred)
r2_test_lr = r2_score(y_test, y_test_pred)

print(f"PCA - Linear Regression - Train MSE: {mse_train_lr}, Train R^2: {r2_train_lr}")
print(f"PCA - Linear Regression - Test MSE: {mse_test_lr}, Test R^2: {r2_test_lr}")

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# 최적의 하이퍼파라미터로 모델 학습
best_params = study.best_params
model = LinearRegression(**best_params)
model.fit(X_train_pca, y_train)

# 예측값 계산
y_train_pred = model.predict(X_train_pca)
y_test_pred = model.predict(X_test_pca)

# MSE와 R^2 계산
mse_train_lr = mean_squared_error(y_train, y_train_pred)
r2_train_lr = r2_score(y_train, y_train_pred)
mse_test_lr = mean_squared_error(y_test, y_test_pred)
r2_test_lr = r2_score(y_test, y_test_pred)

print(f"No PCA - Linear Regression - Train MSE: {mse_train_lr}, Train R^2: {r2_train_lr}")
print(f"No PCA - Linear Regression - Test MSE: {mse_test_lr}, Test R^2: {r2_test_lr}")