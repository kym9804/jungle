import pandas as pd

# 제공된 CSV 파일을 불러옵니다
sample_filtered_data = pd.read_csv('C+GM+M_50matchID_0723.csv')#매치 ID 데이터 파일 넣기 (ID만 있는거)

# 고유한 matchId를 제거하고 그 수를 셉니다
unique_match_ids = sample_filtered_data['match_id'].drop_duplicates()
unique_match_count = unique_match_ids.count()//2500 +1

print(f"고유한 matchId의 수: {unique_match_count}")

# 데이터프레임을 저장할 빈 리스트를 초기화합니다
dataframes = []

# 파일 번호 범위를 반복하면서 각 CSV 파일을 읽어옵니다
for i in range(1, unique_match_count+1):
    file_path = f'C+GM+M_all_match_data_{i}.csv'#sample_timeline_batch어쩌구를 본인 티어 배치 저장된 파일명으로 바꾸기
    try:
        df = pd.read_csv(file_path)
        dataframes.append(df)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        continue

# 모든 데이터프레임을 하나로 병합합니다
if dataframes:
    merged_dataframe = pd.concat(dataframes, ignore_index=True)

    # 병합된 데이터프레임을 새로운 CSV 파일로 저장합니다
    merged_csv_path = 'merged_C+GM+M_all_match.csv'  #원하는 파일명으로 바꾸기
    merged_dataframe.to_csv(merged_csv_path, index=False)

    print(f"병합된 CSV 파일이 저장되었습니다: {merged_csv_path}")
else:
    print("병합된 파일이 없습니다.")