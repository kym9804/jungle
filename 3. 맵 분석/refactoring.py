import pandas as pd

time_file_path = 'LOL_timeline_Combined.csv'
time_data = pd.read_csv(time_file_path, low_memory=False)

match_file_path = 'match_all_ppcomplete_240802.csv'
match_data = pd.read_csv(match_file_path)

# match_data의 필요한 컬럼 선택
match_data = match_data[['matchId', 'participantId', 'championName', 'win', 'teamId']]

# 데이터 병합
merged_data = pd.merge(time_data, match_data, on=['matchId', 'participantId'], how='inner')

# teamcolor 컬럼 추가
merged_data['teamcolor'] = merged_data['teamId'].map({100: 'blue', 200: 'red'})

# 필요한 컬럼들을 두 번째 인덱스부터 삽입
columns_to_insert = ['championName', 'win', 'teamId', 'teamcolor']
for idx, column in enumerate(columns_to_insert):
    merged_data.insert(2 + idx, column, merged_data.pop(column))

merged_data.to_csv('All_tier_LOL_timeline_Combined.csv')
