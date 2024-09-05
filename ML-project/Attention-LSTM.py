import pandas as pd
import numpy as np
from keras.src.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# CSV 파일 로드
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

# RUL 컬럼 추가 함수
def add_rul_column(df):
    max_cycle_per_id = df.groupby('id')['cycle'].max().reset_index()
    max_cycle_per_id.columns = ['id', 'max_cycle']
    df = pd.merge(df, max_cycle_per_id, on='id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop(columns=['max_cycle'])
    return df

# 훈련 데이터에 RUL 컬럼 추가
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

# LSTM을 위한 재구성
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Attention Layer 정의
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],), initializer='random_normal', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = x * a
        return tf.reduce_sum(output, axis=1)

# Attention-LSTM 모델 정의
input_layer = Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]))
lstm_layer = LSTM(100, return_sequences=True)(input_layer)
attention_layer = AttentionLayer()(lstm_layer)
dense_layer_1 = Dense(50, activation='relu')(attention_layer)
dropout_layer = Dropout(0.2)(dense_layer_1)
output_layer = Dense(1, activation='linear')(dropout_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
# EarlyStopping 콜백 정의
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# 모델 훈련
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=2)

# 모델 평가
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=2)


# 테스트 데이터에 대한 RUL 예측
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'Mean Absolute Error: {mse}')
print(f'R2 Score: {r2}')
print(f'Mean Absolute Percentage Error: {mape}%')
# 결과 표시
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 실제 RUL 값과 예측 RUL 값 그래프
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual RUL')
plt.plot(y_pred, label='Predicted RUL')
plt.title('Actual vs Predicted RUL')
plt.xlabel('Sample')
plt.ylabel('RUL')
plt.legend()
plt.show()