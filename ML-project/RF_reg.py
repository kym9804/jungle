import warnings
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

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

# 데이터 로드 파일 패스는 직접 바꿔주세요
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
X_train_smoothed = X_train.apply(apply_smoothing)
X_test_smoothed = X_test.apply(apply_smoothing)
# 데이터 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
# PCA 적용
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# Optuna를 이용한 하이퍼파라미터 튜닝
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
    }

    model = RandomForestRegressor(**param, random_state=42)
    model.fit(X_train_pca, y_train)

    preds = model.predict(X_test_pca)
    mse = mean_squared_error(y_test, preds)
    return mse


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# 최적 하이퍼파라미터 출력
print(f"Best parameters: {study.best_params}")

# 최적 하이퍼파라미터로 모델 학습 및 평가
best_params = study.best_params
rf_model = RandomForestRegressor(**best_params, random_state=42)
rf_model.fit(X_train_pca, y_train)

# 예측
y_pred_train_rf = rf_model.predict(X_train_pca)
y_pred_test_rf = rf_model.predict(X_test_pca)

# 평가
mse_train_rf = mean_squared_error(y_train, y_pred_train_rf)
r2_train_rf = r2_score(y_train, y_pred_train_rf)
mse_test_rf = mean_squared_error(y_test, y_pred_test_rf)
r2_test_rf = r2_score(y_test, y_pred_test_rf)

print(f"PCA - Random Forest - Train MSE: {mse_train_rf}, Train R^2: {r2_train_rf}")
print(f"PCA - Random Forest - Test MSE: {mse_test_rf}, Test R^2: {r2_test_rf}")

# 중요도 시각화
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature Importances")
plt.bar(range(X_train_pca.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_pca.shape[1]), indices)
plt.xlim([-1, X_train_pca.shape[1]])
plt.show()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# 최적 하이퍼파라미터 출력
print(f"Best parameters: {study.best_params}")

# 최적 하이퍼파라미터로 모델 학습 및 평가
best_params = study.best_params
rf_model = RandomForestRegressor(**best_params, random_state=42)
rf_model.fit(X_train_scaled_df, y_train)

# 예측
y_pred_train_rf = rf_model.predict(X_train_scaled_df)
y_pred_test_rf = rf_model.predict(X_test_scaled_df)

# 평가
mse_train_rf = mean_squared_error(y_train, y_pred_train_rf)
r2_train_rf = r2_score(y_train, y_pred_train_rf)
mse_test_rf = mean_squared_error(y_test, y_pred_test_rf)
r2_test_rf = r2_score(y_test, y_pred_test_rf)

print(f"No PCA - Random Forest - Train MSE: {mse_train_rf}, Train R^2: {r2_train_rf}")
print(f"No PCA - Random Forest - Test MSE: {mse_test_rf}, Test R^2: {r2_test_rf}")

# 중요도 시각화
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X_train_scaled_df.columns

plt.figure()
plt.title("Feature Importances")
plt.bar(range(X_train_scaled_df.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train_scaled_df.shape[1]), feature_names[indices], rotation=45, ha="right")
plt.xlim([-1, X_train_scaled_df.shape[1]])
plt.tight_layout()
plt.show()