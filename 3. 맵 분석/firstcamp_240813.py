import pandas as pd
import numpy as np
import ast

# Load the data from the CSV files
file_path = 'All_tier_LOL_timeline_Combined.csv'
data = pd.read_csv(file_path)
squares_ranges_df = pd.read_csv('Large_Squares_Ranges.csv')

# Filter the data where the 'Tier' column is 'C+GM+M'
filtered_data = data

# Define the gold and experience ranges for each jungle monster
jungle_monsters = {
    '블루/레드': {'gold': 90, 'xp': 325},
    '두꺼비': {'gold': 80, 'xp': 350},
    '늑대': {'gold': 85, 'xp': 310},
    '돌거북': {'gold': 109, 'xp': 351},
    '바위게': {'gold': 55, 'xp': 330},
    '칼날부리': {'gold': 75, 'xp': 300}
}

# Calculate the total gold and apply vectorized operations
filtered_data['totalGold'] = filtered_data['totalGold_2']
xp = filtered_data['xp_2']
total_gold = filtered_data['totalGold_2']

conditions = [
    (xp.between(jungle_monsters['블루/레드']['xp'], jungle_monsters['블루/레드']['xp'] + 15)) & (total_gold.between(500 + 90+20, 500 + 90 + 50+20)),
    (xp.between(jungle_monsters['늑대']['xp'], jungle_monsters['늑대']['xp'] + 15)) & (total_gold.between(500 + 85+20, 500 + 85 + 50+20)),
    (xp.between(jungle_monsters['칼날부리']['xp'], jungle_monsters['칼날부리']['xp'] + 15)) & (total_gold.between(500 + 75+20, 500 + 75 + 50+20))
]

choices = ['블루/레드',  '늑대',  '칼날부리']
filtered_data['firstcamp'] = np.select(conditions, choices, default=None)

# Filter out rows where firstcamp is not null
non_null_firstcamp_data = filtered_data[filtered_data['firstcamp'].notnull()].copy()


# Extract x, y values and vectorize the square determination
non_null_firstcamp_data['position_2'] = non_null_firstcamp_data['position_2'].apply(ast.literal_eval)
non_null_firstcamp_data['x'] = non_null_firstcamp_data['position_2'].apply(lambda pos: pos['x'])
non_null_firstcamp_data['y'] = non_null_firstcamp_data['position_2'].apply(lambda pos: pos['y'])

# Only execute this section if 'firstcamp' is '블루/레드'
is_blue_red = non_null_firstcamp_data['firstcamp'] == '블루/레드'

# Use numpy to vectorize the determination of squares for '블루/레드' cases
squares_array = np.zeros(len(non_null_firstcamp_data), dtype=int)
for index, row in squares_ranges_df.iterrows():
    within_x = (non_null_firstcamp_data['x'] >= row['x_start']) & (non_null_firstcamp_data['x'] <= row['x_end'])
    within_y = (non_null_firstcamp_data['y'] >= row['y_start']) & (non_null_firstcamp_data['y'] <= row['y_end'])
    squares_array[within_x & within_y & is_blue_red] = row['Square']

non_null_firstcamp_data.loc[is_blue_red, 'Square'] = squares_array[is_blue_red]

# Vectorize the adjustment of 'firstcamp' based on the area_dict_big2 for '블루/레드' cases
area_dict_big2 = {
    "red team upper jungle": [623, 624, 625, 626, 655, 656, 657, 658, 659, 687, 688, 689, 690, 691, 719, 720, 721, 722, 750, 751, 752, 753, 754, 782, 783, 784, 785, 786, 816, 817, 813, 814, 815, 844, 845, 846, 847, 848, 876, 877, 878, 879, 880, 654, 685, 686, 717, 718, 723, 724, 725, 749, 755, 756, 757, 758, 778, 779, 780, 781, 787, 788, 789, 790, 810, 811, 812, 818, 819, 820, 821, 841, 842, 843, 849, 850, 851, 852, 853, 873, 874, 875, 881, 882, 883, 884, 885],
    "red team lower jungle": [411, 412, 443, 444, 475, 476, 440, 441, 472, 473, 474, 504, 505, 506, 537, 538, 566, 567, 568, 569, 598, 599, 600, 601, 631, 632, 633, 283, 284, 314, 315, 316, 346, 347, 348, 377, 378, 379, 380, 408, 409, 410, 439, 442, 469, 470, 471, 501, 502, 503, 507, 508, 533, 534, 535, 536, 539, 540, 570, 571, 572, 602, 603, 604, 634, 635, 636, 664, 665, 666, 667, 668],
    "blue team lower jungle": [334, 335, 336, 337, 338, 366, 367, 368, 369, 370, 399, 400, 401, 402, 208, 209, 239, 240, 241, 242, 243, 271, 272, 273, 274, 275, 303, 304, 305, 306, 145, 146, 147, 148, 149, 177, 178, 179, 180, 181, 210, 211, 212, 140, 141, 142, 143, 144, 150, 151, 152, 172, 173, 174, 175, 176, 182, 183, 184, 204, 205, 206, 207, 213, 214, 215, 216, 235, 236, 237, 238, 244, 245, 246, 247, 267, 268, 269, 270, 276, 300, 301, 302, 307, 308, 339, 340, 371],
    "blue team upper jungle": [549, 550, 581, 582, 613, 614, 519, 520, 521, 551, 552, 553, 584, 585, 392, 393, 394, 424, 425, 426, 427, 456, 457, 458, 459, 487, 488, 357, 358, 359, 360, 361, 389, 390, 391, 421, 422, 423, 453, 454, 455, 485, 486, 489, 490, 491, 492, 517, 518, 522, 523, 524, 554, 555, 583, 586, 615, 616, 617, 645, 646, 647, 648, 677, 678, 679, 709, 710, 711, 741, 742]
}

red_upper = np.isin(non_null_firstcamp_data['Square'], area_dict_big2['red team upper jungle'])
red_lower = np.isin(non_null_firstcamp_data['Square'], area_dict_big2['red team lower jungle'])
blue_upper = np.isin(non_null_firstcamp_data['Square'], area_dict_big2['blue team upper jungle'])
blue_lower = np.isin(non_null_firstcamp_data['Square'], area_dict_big2['blue team lower jungle'])

non_null_firstcamp_data.loc[is_blue_red & (red_upper | blue_lower), 'firstcamp'] = '레드'
non_null_firstcamp_data.loc[is_blue_red & (red_lower | blue_upper), 'firstcamp'] = '블루'

# Select the relevant columns for the final data
final_data = non_null_firstcamp_data[['Tier', 'matchId', 'championName', 'win', 'teamId', 'firstcamp']]

# Check for duplicates in the final data
final_data = final_data.drop_duplicates(subset=['matchId', 'championName'])

# Check for matchId pairs and filter out any cases where matchId is not duplicated exactly twice
final_data = final_data[final_data['firstcamp'] != '블루/레드']
# Save the final data to a CSV file
final_data.to_csv('First_camp_240813.csv', index=False, encoding="utf-8-sig")

# win 컬럼의 요소 개수 확인
win_count = final_data['win'].value_counts()

# matchId가 2개씩 있는지 확인
match_id_counts = final_data['matchId'].value_counts()
match_id_with_two = match_id_counts[match_id_counts == 2]

# 결과 출력
print("win 컬럼의 요소 개수:\n", win_count)
print("\n2개의 matchId를 가진 matchId의 개수: ", len(match_id_with_two))

# matchId가 2개씩 있는지 여부 확인
if len(match_id_with_two) == len(match_id_counts):
    print("모든 matchId가 2개씩 있습니다.")
else:
    print(f"{len(match_id_counts) - len(match_id_with_two)}개의 matchId가 2개가 아닙니다.")

print("The final data has been processed and saved to 'final_data_adjusted.csv'")
