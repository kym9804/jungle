import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 데이터프레임 불러오기
df = pd.read_csv('merged_df_240802.csv')

# 제외할 컬럼 정의
columns_to_exclude = [
    'Unnamed: 0.1', 'Unnamed: 0', 'matchId', 'gameType',
    'championName_x', 'item0', 'item1', 'item2', 'item3', 'item4',
    'item5', 'item6', 'participantId_x', 'teamId', 'teamPosition',
    'perk1', 'perk1_var1', 'perk1_var2', 'perk1_var3', 'perk2', 'perk2_var1',
    'perk2_var2', 'perk2_var3', 'perk3', 'perk3_var1', 'perk3_var2',
    'perk3_var3', 'perk4', 'perk4_var1', 'perk4_var2', 'perk4_var3',
    'perk5', 'perk5_var1', 'perk5_var2', 'perk5_var3', 'perk6',
    'perk6_var1', 'perk6_var2', 'perk6_var3', 'season',
    'CcontrolWardTimeCoverageInRiverOrEnemyHalf', 'cluster', 'teamcolor',
]

# 지정된 컬럼을 제외한 새로운 데이터프레임 생성
df_filtered = df.drop(columns=columns_to_exclude)

# 'damperdeath' 이후의 시계열 컬럼 식별
time_series_start_index = df_filtered.columns.get_loc('damperdeath') + 1
time_series_columns = df_filtered.columns[time_series_start_index:]

# 'position_'으로 시작하는 컬럼 제외
time_series_columns = [col for col in time_series_columns if not col.startswith('position_')]

# 10분 이하의 컬럼만 추출
time_series_columns_10min = [col for col in time_series_columns if int(col.split('_')[-1]) <= 10]

# 최종 데이터프레임 생성
final_columns = df_filtered.columns[:time_series_start_index].tolist() + time_series_columns_10min
df_final = df_filtered[final_columns]

# 타겟 변수와 특징 변수 정의
target = 'win_x'  # 'win_x'가 타겟 변수라고 가정
features = [col for col in df_final.columns if col != target]
X = df_final[features]
y = df_final[target]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 정의
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 얼리스탑핑 정의 (patience 값을 5로 줄임)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 학습
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping])

# 모델 평가
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# 학습 곡선 시각화
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 각 레이어의 가중치와 편향 출력
for i, layer in enumerate(model.layers):
    weights, biases = layer.get_weights()
    print(f"Layer {i+1} - Weights shape: {weights.shape}, Biases shape: {biases.shape}")
    print(f"Weights: {weights}")
    print(f"Biases: {biases}")
