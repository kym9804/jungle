import requests
import pandas as pd
import json
import time
from tqdm import tqdm

# Read match IDs from the CSV file
data = pd.read_csv('sample_filtered_data.csv')  # 파일 이름 바꾸기

# teamPosition이 JUNGLE인 데이터 필터링
jungle_data = data[data['teamPosition'] == 'JUNGLE']
# 원본 순서대로 matchId와 participantId를 사용하여 딕셔너리 생성
match_dict = {}
for _, row in jungle_data.iterrows():
    match_id = row['matchId']
    participant_id = row['participantId']
    if match_id not in match_dict:
        match_dict[match_id] = []
    if len(match_dict[match_id]) < 2:
        match_dict[match_id].append(participant_id)

# API key
api_key = 'mykey'  # 자기 API키 넣기

# 요청 제한 설정
REQUESTS_PER_SECOND_LIMIT = 19
REQUESTS_PER_TWO_MINUTES_LIMIT = 99
DELAY_BETWEEN_REQUESTS = 1.5 / REQUESTS_PER_SECOND_LIMIT  # 초 단위 지연 설정

# 헤더 설정
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://developer.riotgames.com",
    "X-Riot-Token": api_key  # 실제 API 키로 대체하세요
}

def fetch_timeline_data(match_id):
    timeline_url = f'https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline'
    timeline_response = requests.get(timeline_url, headers=headers)
    if timeline_response.status_code == 200:
        return timeline_response.json()
    else:
        print(f'Failed to retrieve timeline data for {match_id}: {timeline_response.status_code}')
        return None

# 데이터 수집 및 저장 설정
start_time = time.time()
request_count = 0
start_file_number = 0  # 수동으로 시작 파일 번호 설정 (지금까지 저장된 파일 번호 1번까지 저장됐으면 1 넣기 저장된거 없으면 0넣기 뻑났으면 2분정도 쉬어주기)
start_index = (start_file_number) * 2500  # 시작 인덱스 계산

# 매치 아이디당 데이터 호출 및 처리
match_ids = list(match_dict.keys())
for i in tqdm(range(start_index, len(match_ids), 2500), desc="Processing batches", ncols=100, leave=True):  # 매 2500개의 아이디로 나눔
    batch_match_ids = match_ids[i:i + 2500]  # 2500개씩 매치 ID 리스트를 자름
    all_data = []

    for match_id in tqdm(batch_match_ids, desc=f"Processing matches in batch {i // 2500 + 1}", ncols=100, leave=True):
        timeline_data = fetch_timeline_data(match_id)
        request_count += 1

        if timeline_data:
            match_id_from_data = timeline_data['metadata']['matchId']
            for pid in match_dict[match_id]:
                participant_data = {'matchId': match_id_from_data, 'participantId': pid}
                for frame in timeline_data['info']['frames']:
                    timestamp = frame['timestamp'] // 60000
                    frame_data = frame['participantFrames'].get(str(pid), {})
                    if frame_data:
                        participant_data.update({
                            f'currentGold_{timestamp}': frame_data.get('currentGold'),
                            f'jungleMinionsKilled_{timestamp}': frame_data.get('jungleMinionsKilled'),
                            f'level_{timestamp}': frame_data.get('level'),
                            f'position_{timestamp}': frame_data.get('position'),
                            f'totalGold_{timestamp}': frame_data.get('totalGold'),
                            f'xp_{timestamp}': frame_data.get('xp')
                        })
                all_data.append(participant_data)

        # 요청 카운트 증가 및 지연 추가
        time.sleep(DELAY_BETWEEN_REQUESTS)

        # 1초에 20번의 요청 제한을 초과하지 않도록 대기
        if request_count % REQUESTS_PER_SECOND_LIMIT == 0:
            time.sleep(1.2)

        # 2분에 100번의 요청 제한을 초과하지 않도록 대기
        if request_count % REQUESTS_PER_TWO_MINUTES_LIMIT == 0:
            elapsed_time = time.time() - start_time
            if elapsed_time < 120:  # 2분이 지나지 않았다면 대기
                time.sleep(120 - elapsed_time + 1)
            start_time = time.time()  # 시작 시간 갱신

    # 데이터프레임 생성 및 저장
    df = pd.DataFrame(all_data)
    file_number = (i // 2500) + 1  # 파일 번호 계산
    df.to_csv(f'sample_timeline_batch_{file_number}.csv', index=False)  # 파일 저장
    print(f"데이터가 성공적으로 저장되었습니다: sample_timeline_batch_{file_number}.csv")

