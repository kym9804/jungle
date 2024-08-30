import pandas as pd
import matplotlib.pyplot as plt
import ast

# 데이터 로드
file_path = 'merged_C+GM+M_timeline.csv'
data = pd.read_csv(file_path, low_memory=False)

# 'position_{minute}' 컬럼 필터링
position_columns = [col for col in data.columns if col.startswith('position_')]


# 문자열 파싱 함수
def parse_position(position_str):
    try:
        position_dict = ast.literal_eval(position_str)
        return position_dict['x'], position_dict['y']
    except (ValueError, SyntaxError, KeyError):
        return None, None


# minute을 선택하여 히트맵 그리기
def plot_heatmap(minute):
    column_name = f'position_{minute}'

    if column_name not in position_columns:
        raise ValueError(f"No data available for minute {minute}")

    # 데이터 파싱
    positions = data[column_name].dropna().apply(parse_position)
    positions = pd.DataFrame(positions.tolist(), columns=['x', 'y']).dropna()

    x = positions['x']
    y = positions['y']

    plt.figure(figsize=(16, 16))
    plt.hist2d(x, y, bins=[range(0, 16001, 500), range(0, 16001, 500)], cmap='hot')
    plt.colorbar(label='Density')
    plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(range(0, 16001, 500))
    plt.yticks(range(0, 16001, 500))
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Heatmap for Minute {minute}')
    plt.show()


# 예시: 10분에 해당하는 히트맵 그리기
plot_heatmap(2)
