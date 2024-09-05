import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# 예제 데이터 생성
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [2, 3, 4, 5, 6],
    'Feature3': [3, 4, 5, 6, 7]
}
df = pd.DataFrame(data)

# PCA 모델 생성 및 데이터에 적용
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df)

print(pca.feature_names_in_)
