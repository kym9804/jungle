import json
import time
import requests
import pandas as pd
from tqdm import tqdm
API = 'mykey'#API키 넣기

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://developer.riotgames.com",
    "X-Riot-Token": API
}
df = pd.read_csv('D_summonerId.csv')#파일에 맞게 이름 바꾸기

puuid_list = df['puuid'].unique().tolist()
# Extract unique puuid
print(f"Unique PUUID count: {len(puuid_list)}")
# Initialize an empty set to keep track of unique match IDs
unique_match_ids = set()

# Function to fetch match IDs for a given puuid
def fetch_match_ids(puuid):
    url = f"https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?type=ranked&start=0&count=40"#여기 count 수 수정
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return []

# Iterate through each puuid and fetch match IDs
for puuid in tqdm(puuid_list, desc="Fetching match IDs"):
    match_ids = fetch_match_ids(puuid)
    unique_match_ids.update(match_ids)

# Convert the set of unique match IDs to a list
unique_match_ids = list(unique_match_ids)

# Save the unique match IDs to a CSV file
match_ids_df = pd.DataFrame(unique_match_ids, columns=['match_id'])
match_ids_df.to_csv('LOL_matchID_D.csv', index=False)

print("매치 ID CSV 파일이 성공적으로 저장되었습니다.")
