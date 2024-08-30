import requests
import pandas as pd
import json
import time
from tqdm import tqdm

# Read match IDs from the CSV file
match_ids_df = pd.read_csv('LOL_matchID_D.csv')#파일 이름 바꾸기
match_id_list = match_ids_df['match_id'].tolist()
# API key
api_key = 'mykey'  # 자기 API키 넣기

# 요청 제한 설정
REQUESTS_PER_SECOND_LIMIT = 20
REQUESTS_PER_TWO_MINUTES_LIMIT = 100
DELAY_BETWEEN_REQUESTS = 1.5 / REQUESTS_PER_SECOND_LIMIT  # 초 단위 지연 설정

# 매치 아이디 리스트 로드

# 헤더 설정
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Charset": "application/x-www-form-urlencoded; charset=UTF-8",
    "Origin": "https://developer.riotgames.com",
    "X-Riot-Token": api_key  # 실제 API 키로 대체하세요
}

# 데이터 프레임 초기화
filtered_data = []

# 시작 시간 및 요청 카운트 초기화
start_time = time.time()
request_count = 0

# 매치 아이디당 데이터 호출 및 처리
# 매치 아이디당 데이터 호출 및 처리
for match_id in tqdm(match_id_list, desc="Processing matches"):
    url = f'https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}'
    response = requests.get(url, headers=headers)


    # 요청 성공 여부 확인
    if response.status_code == 200:
        data = response.json()
        info = data['info']
        if info['gameMode'] == "CLASSIC" and info['mapId'] == 11:
            mode = info['gameMode']
            map = info['mapId']
            metadata = data['metadata']

            # 메타데이터 및 플레이어 정보 추출
            matchId = metadata['matchId']
            gameVersion = info['gameVersion']
            participants = info['participants']

            # 플레이어 데이터 필터링
            for participant in participants:
                challenges = participant['challenges']
                perks = participant['perks']
                statPerks = perks['statPerks']
                primaryStyle = perks['styles'][0]['selections']
                subStyle = perks['styles'][1]['selections']
                row = {
                    'matchId': matchId,
                    'gameType': gameVersion,
                    'assists': participant.get('assists'),
                    'baronKills': participant.get('baronKills'),
                    'champExperience': participant.get('champExperience'),
                    'champLevel': participant.get('champLevel'),
                    'championName': participant.get('championName'),
                    'damageDealtToBuildings': participant.get('damageDealtToBuildings'),
                    'damageDealtToObjectives': participant.get('damageDealtToObjectives'),
                    'damageDealtToTurrets': participant.get('damageDealtToTurrets'),
                    'damageSelfMitigated': participant.get('damageSelfMitigated'),
                    'deaths': participant.get('deaths'),
                    'detectorWardsPlaced': participant.get('detectorWardsPlaced'),
                    'dragonKills': participant.get('dragonKills'),
                    'goldEarned': participant.get('goldEarned'),
                    'inhibitorTakedowns': participant.get('inhibitorTakedowns'),
                    'item0': participant.get('item0'),
                    'item1': participant.get('item1'),
                    'item2': participant.get('item2'),
                    'item3': participant.get('item3'),
                    'item4': participant.get('item4'),
                    'item5': participant.get('item5'),
                    'item6': participant.get('item6'),
                    'kills': participant.get('kills'),
                    'magicDamageDealtToChampions': participant.get('magicDamageDealtToChampions'),
                    'magicDamageTaken': participant.get('magicDamageTaken'),
                    'neutralMinionsKilled': participant.get('neutralMinionsKilled'),
                    'objectivesStolen': participant.get('objectivesStolen'),
                    'objectivesStolenAssists': participant.get('objectivesStolenAssists'),
                    'physicalDamageDealtToChampions': participant.get('physicalDamageDealtToChampions'),
                    'physicalDamageTaken': participant.get('physicalDamageTaken'),
                    'participantId': participant['participantId'],
                    'teamId': participant.get('teamId'),
                    'teamPosition': participant.get('teamPosition'),
                    'timeCCingOthers': participant.get('timeCCingOthers'),
                    'timePlayed': participant.get('timePlayed'),
                    'totalAllyJungleMinionsKilled': participant.get('totalAllyJungleMinionsKilled'),
                    'totalDamageDealtToChampions': participant.get('totalDamageDealtToChampions'),
                    'totalDamageShieldedOnTeammates': participant.get('totalDamageShieldedOnTeammates'),
                    'totalDamageTaken': participant.get('totalDamageTaken'),
                    'totalEnemyJungleMinionsKilled': participant.get('totalEnemyJungleMinionsKilled'),
                    'totalHeal': participant.get('totalHeal'),
                    'totalHealsOnTeammates': participant.get('totalHealsOnTeammates'),
                    'totalMinionsKilled': participant.get('totalMinionsKilled'),
                    'totalTimeSpentDead': participant.get('totalTimeSpentDead'),
                    'turretTakedowns': participant.get('turretTakedowns'),
                    'visionScore': participant.get('visionScore'),
                    'win': participant.get('win'),
                    'alliedJungleMonsterKills': challenges.get('alliedJungleMonsterKills'),
                    'baronTakedowns': challenges.get('baronTakedowns'),
                    'controlWardTimeCoverageInRiverOrEnemyHalf': challenges.get(
                        'controlWardTimeCoverageInRiverOrEnemyHalf'),
                    'damageTakenOnTeamPercentage': challenges.get('damageTakenOnTeamPercentage'),
                    'enemyJungleMonsterKills': challenges.get('enemyJungleMonsterKills'),
                    'epicMonsterKillsNearEnemyJungler': challenges.get('epicMonsterKillsNearEnemyJungler'),
                    'epicMonsterKillsWithin30SecondsOfSpawn': challenges.get('epicMonsterKillsWithin30SecondsOfSpawn'),
                    'epicMonsterSteals': challenges.get('epicMonsterSteals'),
                    'initialCrabCount': challenges.get('initialCrabCount'),
                    'jungleCsBefore10Minutes': challenges.get('jungleCsBefore10Minutes'),
                    'killsNearEnemyTurret': challenges.get('killsNearEnemyTurret'),
                    'killsOnLanersEarlyJungleAsJungler': challenges.get('killsOnLanersEarlyJungleAsJungler'),
                    'laningPhaseGoldExpAdvantage': challenges.get('laningPhaseGoldExpAdvantage'),
                    'maxLevelLeadLaneOpponent': challenges.get('maxLevelLeadLaneOpponent'),
                    'moreEnemyJungleThanOpponent': challenges.get('moreEnemyJungleThanOpponent'),
                    'pickKillWithAlly': challenges.get('pickKillWithAlly'),
                    'scuttleCrabKills': challenges.get('scuttleCrabKills'),
                    'soloKills': challenges.get('soloKills'),
                    'takedownsFirstXMinutes': challenges.get('takedownsFirstXMinutes'),
                    'takedownsInAlcove': challenges.get('takedownsInAlcove'),
                    'teamDamagePercentage': challenges.get('teamDamagePercentage'),
                    'visionScoreAdvantageLaneOpponent': challenges.get('visionScoreAdvantageLaneOpponent')
                }
                for i, perk in enumerate(primaryStyle + subStyle):
                    row[f'perk{i + 1}'] = perk['perk']
                    row[f'perk{i + 1}_var1'] = perk['var1']
                    row[f'perk{i + 1}_var2'] = perk['var2']
                    row[f'perk{i + 1}_var3'] = perk['var3']
                filtered_data.append(row)
    else:
        print(f"요청 실패, 상태 코드: {response.status_code}, 매치 아이디: {match_id}")

    # 요청 카운트 증가 및 지연 추가
    request_count += 1
    time.sleep(DELAY_BETWEEN_REQUESTS)

    # 1초에 20번의 요청 제한을 초과하지 않도록 대기
    if request_count % (REQUESTS_PER_SECOND_LIMIT-1) == 0:
        sleep_time = 1 - DELAY_BETWEEN_REQUESTS * REQUESTS_PER_SECOND_LIMIT
        if sleep_time > 0:
            time.sleep(sleep_time + 1)

    # 2분에 100번의 요청 제한을 초과하지 않도록 대기
    if request_count % (REQUESTS_PER_TWO_MINUTES_LIMIT-1) == 0:
        elapsed_time = time.time() - start_time
        if elapsed_time < 120:  # 2분이 지나지 않았다면 대기
            time.sleep(120 - elapsed_time +1)
        start_time = time.time()  # 시작 시간 갱신

# 데이터프레임 생성 및 저장
df = pd.DataFrame(filtered_data)
df.to_csv('sample_filtered_data2.csv', index=False)#파일이름 바꾸기
print("데이터가 성공적으로 저장되었습니다.")
