import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from tqdm import tqdm
import matplotlib.image as mpimg

# Load the data
data_file_path = 'All_tier_LOL_timeline_Combined.csv'
data = pd.read_csv(data_file_path, low_memory=False)

squares_file_path = 'Large_Squares_Ranges.csv'
squares_data = pd.read_csv(squares_file_path)

# Function to parse position strings
def parse_position(position_str):
    try:
        position_dict = ast.literal_eval(position_str)
        return position_dict['x'], position_dict['y']
    except (ValueError, SyntaxError, KeyError):
        return None, None

# Placeholder for area_dict
area_dict = {
    "top lane": [353, 354, 355, 356, 385, 386, 387, 388, 417, 418, 419, 420, 449, 450, 451, 452, 481, 482, 483, 484, 513, 514, 515, 516, 545, 546, 547, 548, 577, 578, 579, 580, 609, 610, 611, 612, 641, 642, 643, 644, 673, 674, 675, 676, 705, 706, 707, 708, 737, 738, 739, 740, 769, 770, 771, 772, 773, 774, 801, 802, 803, 804, 805, 806, 807, 808, 833, 834, 835, 836, 837, 838, 839, 840, 870, 871, 872, 902, 903, 904, 905, 906, 907, 908, 909, 910, 911, 912, 913, 914, 915, 916, 917, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 998, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013],
    "top alcove": [865, 866, 867, 868, 869, 897, 898, 899, 900, 901, 929, 930, 931, 932, 933, 961, 962, 963, 964, 965, 993, 994, 995, 996, 997],
    "red team upper jungle": [654, 685, 686, 717, 718, 723, 724, 725, 749, 755, 756, 757, 758, 778, 779, 780, 781, 787, 788, 789, 790, 810, 811, 812, 818, 819, 820, 821, 841, 842, 843, 849, 850, 851, 852, 853, 873, 874, 875, 881, 882, 883, 884, 885],
    "red team krug": [813, 814, 815, 844, 845, 846, 847, 848, 876, 877, 878, 879, 880],
    "red team red": [719, 720, 721, 722, 750, 751, 752, 753, 754, 782, 783, 784, 785, 786, 816, 817],
    "red team raptor": [623, 624, 625, 626, 655, 656, 657, 658, 659, 687, 688, 689, 690, 691],
    "red team home": [698, 699, 700, 701, 702, 703, 704, 728, 729, 730, 731, 732, 733, 734, 735, 736, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 982, 983, 984, 985, 986, 987, 988, 989, 990, 991, 992, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024],
    "red team wolf": [537, 538, 566, 567, 568, 569, 598, 599, 600, 601, 631, 632, 633],
    "red team lower jungle": [283, 284, 314, 315, 316, 346, 347, 348, 377, 378, 379, 380, 408, 409, 410, 439, 442, 469, 470, 471, 501, 502, 503, 507, 508, 533, 534, 535, 536, 539, 540, 570, 571, 572, 602, 603, 604, 634, 635, 636, 664, 665, 666, 667, 668],
    "red team blue": [440, 441, 472, 473, 474, 504, 505, 506],
    "red team gromp": [411, 412, 443, 444, 475, 476],
    "dragon river": [248, 249, 250, 280, 281, 282, 312, 313, 344, 345, 372, 373, 374, 375, 376, 403, 404, 405, 406, 407, 435, 436, 437, 438, 468],
    "dragon nest": [277, 278, 279, 309, 310, 311, 341, 342, 343],
    "blue team lower jungle": [140, 141, 142, 143, 144, 150, 151, 152, 172, 173, 174, 175, 176, 182, 183, 184, 204, 205, 206, 207, 213, 214, 215, 216, 235, 236, 237, 238, 244, 245, 246, 247, 267, 268, 269, 270, 276, 300, 301, 302, 307, 308, 339, 340, 371],
    "bottom lane": [12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,153,154,155,185,186,187,188,189,190,191,192,217,218,219,220,221,222,223,224,251,252,253,254,255,256,285,286,287,288,317,318,319,320,349,350,351,352,381,382,383,384,413,414,415,416,445,446,447,448,477,478,479,480,509,510,511,512,541,542,543,544,573,574,575,576,605,606,607,608,637,638,639,640,669,670,671,672],
    "bottom alcove":[28,29,30,31,32,60,61,62,63,64,92,93,94,95,96,124,125,126,127,128,156,157,158,159,160],
    "blue team krug":[145,146,147,148,149,177,178,179,180,181,210,211,212],
    "blue team red":[208,209,239,240,241,242,243,271,272,273,274,275,303,304,305,306],
    "blue team raptor":[334,335,336,337,338,366,367,368,369,370,399,400,401,402],
    "mid lane":[298,299,328,329,330,331,332,333,362,363,364,365,395,396,397,398,428,429,430,431,432,433,434,460,461,462,463,464,465,466,467,493,494,495,496,497,498,499,500,525,526,527,528,529,530,531,532,558,559,560,561,562,563,564,565,591,592,593,594,595,596,597,627,628,629,630,660,661,662,663,692,693,694,695,696,697,726,727,298, 299, 328, 329, 330, 331, 332, 333, 362, 363, 364, 365, 395, 396, 397, 398, 428, 429, 430, 431, 432, 433, 434, 460, 461, 462, 463, 464, 465, 466, 467, 493, 494, 495, 496, 497, 498, 499, 500, 525, 526, 527, 528, 529, 530, 531, 532, 558, 559, 560, 561, 562, 563, 564, 565, 591, 592, 593, 594, 595, 596, 597, 627, 628, 629, 630, 660, 661, 662, 663, 692, 693, 694, 695, 696, 697, 726, 72],
    "blue team home":[1,2,3,4,5,6,7,8,9,10,11,33,34,35,36,37,38,39,40,41,42,43,65,66,67,68,69,70,71,72,73,74,75,97,98,99,100,101,102,103,104,105,106,107,129,130,131,132,133,134,135,136,137,138,139,161,162,163,164,165,166,167,168,169,170,171,193,194,195,196,197,198,199,200,201,202,203,225,226,227,228,229,230,231,232,233,234,257,258,259,260,261,262,263,264,265,266,289,290,291,292,293,294,295,296,297,321,322,323,324,325,326,327],
    "blue team upper jungle":[357,358,359,360,361,389,390,391,421,422,423,453,454,455,485,486,489,490,491,492,517,518,522,523,524,554,555,583,586,615,616,617,645,646,647,648,677,678,679,709,710,711,741,742],
    "blue team wolf":[392, 393, 394, 424, 425, 426, 427, 456, 457, 458, 459, 487, 488],
    "blue team blue":[519,520,521,551,552,553,584,585],
    "blue team gromp":[549, 550, 581, 582, 613, 614],
    "baron river":[556,557,587,588,589,590,618,619,620,621,622,649,650,651,652,653,680,681,712,713,743,744,745,775,776,777,809],
    "baron nest":[682,683,684,714,715,716,746,747,748]
}


# Reverse the dictionary to map each square number to its corresponding area
square_to_area = {square: area for area, squares in area_dict.items() for square in squares}

# Function to find area from square number
def find_area(square, square_to_area):
    return square_to_area.get(square, None)

# Mock function to replace find_square_number_vectorized
def find_square_number_vectorized(positions, squares_data):
    x = positions['x'].values
    y = positions['y'].values

    square_numbers = np.full(len(x), None)

    for index, row in squares_data.iterrows():
        mask = (row['x_start'] <= x) & (x <= row['x_end']) & (row['y_start'] <= y) & (y <= row['y_end'])
        square_numbers[mask] = row['Square']

    return square_numbers
# Redefine the plot_heatmap function to include the background image
# 백그라운드 이미지와 챔피언 필터, 티어 필터를 포함한 히트맵을 그리는 함수
def plot_heatmap_with_background(minute, background_image, champion_name=None, tier=None):
    column_name = f'position_{minute}'

    if column_name not in data.columns:
        raise ValueError(f"{minute}분에 대한 데이터가 없습니다")

    # 지정된 챔피언과 티어에 대한 데이터 필터링
    filtered_data = data
    if champion_name:
        filtered_data = filtered_data[filtered_data['championName'] == champion_name]
    if tier:
        filtered_data = filtered_data[filtered_data['Tier'] == tier]

    # 데이터 파싱
    positions = filtered_data[column_name].dropna().apply(parse_position)
    positions = pd.DataFrame(positions.tolist(), columns=['x', 'y']).dropna()

    # 벡터화된 함수로 사각형 번호 추가
    tqdm.pandas(desc="사각형 찾는 중")
    positions['Square'] = find_square_number_vectorized(positions, squares_data)

    # 위치 데이터의 첫 몇 행을 출력
    print(positions.head())

    x = positions['x']
    y = positions['y']

    plt.figure(figsize=(16, 16))
    plt.imshow(background_image, extent=[0, 16000, 0, 16000], aspect='auto')
    plt.hist2d(x, y, bins=[range(0, 16001, 500), range(0, 16001, 500)], cmap='hot', alpha=0.6)
    plt.colorbar(label='밀도')
    plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.xticks(range(0, 16001, 500))
    plt.yticks(range(0, 16001, 500))
    plt.xlabel('X 위치')
    plt.ylabel('Y 위치')
    title = f'{minute}분의 히트맵'
    if champion_name:
        title += f' ({champion_name})'
    if tier:
        title += f' [Tier: {tier}]'
    plt.title(title)
    plt.show()

# 백그라운드 이미지 로드
background_image_path = '협곡 이미지_32×32 2.png'
background_image = mpimg.imread(background_image_path)

# 예시: 특정 챔피언과 티어에 대한 3분 히트맵 그리기
plot_heatmap_with_background(3, background_image, champion_name='LeeSin', tier='Gold')