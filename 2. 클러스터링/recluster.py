import pandas as pd
import numpy as np
from itertools import combinations, islice
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

# 데이터 불러오기
data_file_path = 'LOL_match_CGM_ppcomplete_240807.csv'
df = pd.read_csv(data_file_path)

# 제외할 컬럼들
columns_to_exclude = [
    'Tier', 'matchId', 'gameType', 'championName', 'item0', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6',
    'participantId', 'teamId', 'teamPosition', 'win', 'perk1', 'perk1_var1', 'perk1_var2', 'perk1_var3', 'perk2',
    'perk2_var1',
    'perk2_var2', 'perk2_var3', 'perk3', 'perk3_var1', 'perk3_var2', 'perk3_var3', 'perk4', 'perk4_var1', 'perk4_var2',
    'perk4_var3', 'perk5', 'perk5_var1', 'perk5_var2', 'perk5_var3', 'perk6', 'perk6_var1', 'perk6_var2', 'perk6_var3',
    'season'
]

# 제외할 컬럼들 드롭
df_filtered = df.drop(columns=columns_to_exclude)


def evaluate_combination(combo):
    selected_columns = list(combo)
    df_subset = df_filtered[selected_columns]

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_subset)

    pca = PCA(n_components=3)
    df_pca = pca.fit_transform(df_scaled)

    results = []
    for n_clusters in [3, 4]:
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(df_pca)
        silhouette_avg = silhouette_score(df_pca, cluster_labels)

        if silhouette_avg >= 0.5:
            results.append((selected_columns, n_clusters, silhouette_avg))

    return results


def main():
    columns = df_filtered.columns
    num_workers = 6
    chunk_size = 10000
    if not os.path.exists('output'):
        os.makedirs('output')

    total_combinations = sum(1 for _ in combinations(columns, 13))

    for start in range(0, total_combinations, chunk_size):
        end = min(start + chunk_size, total_combinations)
        combinations_chunk = list(islice(combinations(columns, 13), start, end))

        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(evaluate_combination, combinations_chunk), total=len(combinations_chunk),
                                desc=f'Processing chunk {start // chunk_size + 1}'))

        flat_results = [item for sublist in results for item in sublist]

        if flat_results:
            results_df = pd.DataFrame(flat_results, columns=['Columns', 'Num_Clusters', 'Silhouette_Score'])
            results_df.to_csv(f'columns_set_{start // chunk_size + 1}.csv', index=False)


if __name__ == '__main__':
    main()
