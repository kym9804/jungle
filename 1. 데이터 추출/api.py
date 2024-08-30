import json
import time
import requests
import pandas as pd
from tqdm import tqdm
REQUESTS_PER_SECOND_LIMIT = 20
REQUESTS_PER_TWO_MINUTES_LIMIT = 100
DELAY_BETWEEN_REQUESTS = 1.2 / REQUESTS_PER_SECOND_LIMIT  # 1초 제한을 기준으로 설정 (초 단위)

API = 'mykey'#API키 넣기
# API endpoint
urls = [
    'https://kr.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/DIAMOND/I',
    'https://kr.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/DIAMOND/II',
    'https://kr.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/DIAMOND/III',
    'https://kr.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/DIAMOND/IV'
]

# 요청 헤더
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://developer.riotgames.com",
    "X-Riot-Token": API
}
# 요청 수와 시작 시간 추적
request_count = 0
start_time = time.time()
# 모든 데이터를 저장할 리스트
all_entries = []

# 각 URL에서 데이터를 가져와서 리스트에 추가
# Fetch data from each URL
for url in urls:
    time.sleep(1)
    start_index = 1
    total_filtered_entries = 0
    progress_bar = tqdm(total=2500, desc=f'Processing {url}')
    request_count = 0

    while total_filtered_entries < 2500:
        response = requests.get(f'{url}?page={start_index}', headers=headers)
        time.sleep(1)

        # Check if the response is successful
        if response.status_code != 200:
            print(f'Error: Received status code {response.status_code} for URL {url}')
            break

        # Convert the response to JSON format
        try:
            response_json = response.json()
        except json.JSONDecodeError:
            print(f'Error: Failed to decode JSON response for URL {url}')
            break

        # Check if the response data is empty
        if not response_json:
            print(f'No more data available for URL {url} on page {start_index}')
            break

        # Filter out 'inactive' entries
        filtered_entries = []
        with tqdm(total=len(response_json), desc=f'Processing entries in {url} page {1}') as entry_progress:
            for entry in response_json:
                if not entry.get('inactive', False):
                    summoner_id = entry['summonerId']
                    summoner_url = f'https://kr.api.riotgames.com/lol/summoner/v4/summoners/{summoner_id}'
                    summoner_response = requests.get(summoner_url, headers=headers)
                    request_count += 1
                    if summoner_response.status_code == 200:
                        summoner_data = summoner_response.json()
                        if 'puuid' in summoner_data and summoner_data['puuid']:
                            entry['puuid'] = summoner_data['puuid']
                            filtered_entries.append(entry)

                    # 요청 사이에 지연 추가
                    time.sleep(DELAY_BETWEEN_REQUESTS)

                    # 1초에 20번의 요청 제한을 초과하지 않도록 대기
                    if request_count % REQUESTS_PER_SECOND_LIMIT == 0:
                        sleep_time = 1 - DELAY_BETWEEN_REQUESTS * REQUESTS_PER_SECOND_LIMIT
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                    # 2분에 100번의 요청 제한을 초과하지 않도록 대기
                    if request_count % REQUESTS_PER_TWO_MINUTES_LIMIT == 0:
                        elapsed_time = time.time() - start_time
                        if elapsed_time < 120:  # 2분이 지나지 않았다면 대기
                            time.sleep(120 - elapsed_time)
                        start_time = time.time()  # 시작 시간 갱신

                entry_progress.update(1)

        all_entries.extend(filtered_entries)
        total_filtered_entries += len(filtered_entries)
        progress_bar.update(len(filtered_entries))

        # Move to the next page
        start_index += 1

        # Stop if we have collected 2500 filtered entries
        if total_filtered_entries >= 2500:
            time.sleep(1)
            break

        # Add delay between main requests
        time.sleep(1)

    progress_bar.close()

# 데이터를 DataFrame으로 변환
df = pd.DataFrame(all_entries)
df.to_csv('D_summonerId.csv', index=False)#티어이름 바꾸기
time.sleep(1)
# summonerId를 통한 추가 데이터 추출
# 필요한 데이터를 저장할 딕셔너리 초기화
