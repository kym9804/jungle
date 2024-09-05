import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


# RUL 컬럼 추가 함수 정의
def add_rul_column(df):
    max_cycle_per_id = df.groupby('id')['cycle'].max().reset_index()
    max_cycle_per_id.columns = ['id', 'max_cycle']
    df = pd.merge(df, max_cycle_per_id, on='id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop(columns=['max_cycle'])
    return df

# 제공된 CSV 파일 로드
train_FD001_path = 'train_FD001.csv'
test_FD001_path = 'test_FD001.csv'
RUL_FD001_path = 'RUL_FD001.csv'

# 데이터를 pandas DataFrame으로 로드
df_train_FD001 = pd.read_csv(train_FD001_path, header=None)
df_test_FD001 = pd.read_csv(test_FD001_path, header=None)
df_RUL_FD001 = pd.read_csv(RUL_FD001_path, header=None)

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
pre_df_train_FD001 = df_train_FD001.drop(columns=['id', 'setting1', 'setting2', 'setting3'])
pre_df_test_FD001 = df_test_FD001.drop(columns=['id', 'setting1', 'setting2', 'setting3'])

# 특징과 목표 설정
X_train = pre_df_train_FD001.drop(columns=['RUL'])
y_train = pre_df_train_FD001['RUL']
X_test = pre_df_test_FD001.drop(columns=['RUL'])
y_test = pre_df_test_FD001['RUL']
y_test = y_test.clip(upper=125)  # RUL 값을 상한선으로 클리핑
y_train = y_train.clip(upper=125)

# 데이터 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 표준화된 데이터를 다시 DataFrame으로 변환
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# 입력 데이터를 [samples, time steps, features] 형태로 재구성
X_train_reshaped = X_train_scaled_df.values.reshape((X_train_scaled_df.shape[0], X_train_scaled_df.shape[1], 1))
X_test_reshaped = X_test_scaled_df.values.reshape((X_test_scaled_df.shape[0], X_test_scaled_df.shape[1], 1))

# CNN-LSTM 모델 구축
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 조기 종료 콜백
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 학습
history = model.fit(X_train_reshaped, y_train, epochs=100, batch_size=64, validation_data=(X_test_reshaped, y_test), verbose=2, shuffle=False, callbacks=[early_stopping])

# 모델 평가
mse = model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f'Mean Squared Error: {mse}')

# RUL 예측
y_pred = model.predict(X_test_reshaped)

# MAE 및 R² 점수 계산
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'Mean Absolute Error: {mse}')
print(f'R2 Score: {r2}')
print(f'Mean Absolute Percentage Error: {mape}%')

# 학습 및 검증 손실 값 플롯
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# 실제 y값과 예측 y값의 산점도 그리기
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual ROAS')
plt.ylabel('Predicted ROAS')
plt.title('Actual vs Predicted ROAS')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # 실제 값과 예측 값이 일치하는 선
plt.show()