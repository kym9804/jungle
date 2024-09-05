
import warnings
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.linear_model import Lasso

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

def load_data(train_path, test_path, rul_path):
    # 데이터 로드
    df_train = pd.read_csv(train_path, header=None)
    df_test = pd.read_csv(test_path, header=None)
    df_RUL = pd.read_csv(rul_path, header=None)

    sensor_names = [f'sensor{i}' for i in range(1, 22)]
    column_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + sensor_names
    df_train.columns = column_names
    df_test.columns = column_names

    return df_train, df_test, df_RUL

def preprocess_data_weak(df_train, df_test, sensor_names, threshold=0.4, n_components=5):
    # sensor_names와 RUL 간의 상관관계 계산
    correlation_matrix = df_train[['cycle'] + sensor_names + ['RUL']].corr()
    # RUL과의 상관계수만 추출
    correlation_with_RUL = correlation_matrix['RUL']
    # 상관계수가 threshold 이상인 센서 필터링
    high_correlation_sensors = correlation_with_RUL[correlation_with_RUL.abs() >= threshold]
    selected_features = high_correlation_sensors.index.tolist()

    # 필요한 컬럼만 추출
    pre_df_train = df_train[selected_features]
    pre_df_test = df_test[selected_features]

    X_train = pre_df_train.drop(columns=['RUL'])
    y_train = pre_df_train['RUL'].clip(upper=125)
    X_test = pre_df_test.drop(columns=['RUL'])
    y_test = pre_df_test['RUL'].clip(upper=125)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # PCA 적용
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled_df)
    X_test_pca = pca.transform(X_test_scaled_df)

    return X_train_pca, X_test_pca, y_train, y_test

# 사용 예시:
train_path = 'train_FD001.csv'
test_path = 'test_FD001.csv'
rul_path = 'RUL_FD001.csv'

# 센서 이름 정의
sensor_names = [f'sensor{i}' for i in range(1, 22)]

df_train, df_test, df_RUL = load_data(train_path, test_path, rul_path)

df_train = add_rul_column(df_train)
df_RUL.columns = ['RUL']
df_test = df_test.groupby('id').last().reset_index()
df_test['RUL'] = df_RUL['RUL']

X_train_weak, X_test_weak, y_train_weak, y_test_weak = preprocess_data_weak(df_train, df_test, sensor_names)

