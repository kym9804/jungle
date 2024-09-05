import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# 상관계수 행렬 계산
corr_train = df_train_FD001.corr()
corr_test = df_test_FD001.corr()

# 히트맵 그리기
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
sns.heatmap(corr_train, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap for Training Data')

plt.subplot(1, 2, 2)
sns.heatmap(corr_test, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap for Test Data')

plt.tight_layout()
plt.show()
