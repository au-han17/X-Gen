import json
import pandas as pd 
import csv
import numpy as np
def format_string(s):
    # Remove any whitespace
    s = s.replace(" ", "")
    
    # Replace '.' with ','
    s = s.replace(".", ",")
    
    # Split string into separate characters
    chars = list(s)
    
    # Check if there are multiple digits together
    # If yes, then separate them
    for i in range(len(chars)):
        if chars[i].isdigit() and i+1 < len(chars) and chars[i+1].isdigit():
            chars[i] = chars[i] + ','
    
    # Join the characters back into a string
    s = ''.join(chars)
    
    # Remove unwanted characters
    s = ''.join(filter(lambda x: x.isdigit() or x == ',', s))
    
    # Split the string by ','
    split_s = s.split(",")
    
    # Keep only the first four numbers
    split_s = split_s[:4]

    # Join the list back into a string with ',' as separator
    formatted_s = ",".join(split_s)

    # If the formatted string is empty, return the original string
    if formatted_s == '':
        return s

    return formatted_s


index_file = 'IU_val_test_shuffled_all.csv'
index_file_df = pd.read_csv(index_file, skiprows=1)

# List all the file names
other_files = [
    'IU_task2_test3_split5_3.csv', 'IU_task2_test1_split1_1.csv', 'IU_task2_test3_split6_3.csv',
    'IU_task2_test1_split2_1.csv', 'IU_task2_test3_split7_3.csv', 'IU_task2_test1_split3_1.csv',
    'IU_task2_test3_split8_3.csv', 'IU_task2_test1_split4_1.csv', 'IU_task2_val1_split1_1.csv',
    'IU_task2_test1_split5_1.csv', 'IU_task2_val1_split2_1.csv', 'IU_task2_test1_split6_1.csv',
    'IU_task2_val1_split3_1.csv', 'IU_task2_test1_split7_1.csv', 'IU_task2_val1_split4_1.csv',
    'IU_task2_test1_split8_1.csv', 'IU_task2_val1_split5_1.csv', 'IU_task2_test2_split1_2.csv',
    'IU_task2_val1_split6_1.csv', 'IU_task2_test2_split2_2.csv', 'IU_task2_val1_split7_1.csv',
    'IU_task2_test2_split3_2.csv', 'IU_task2_val1_split8_1.csv', 'IU_task2_test2_split4_2.csv',
    'IU_task2_val2_split1_2.csv', 'IU_task2_test2_split5_2.csv', 'IU_task2_val2_split2_2.csv',
    'IU_task2_test2_split6_2.csv', 'IU_task2_val2_split3_2.csv', 'IU_task2_test2_split7_2.csv',
    'IU_task2_val2_split4_2.csv', 'IU_task2_test2_split8_2.csv', 'IU_task2_val2_split5_2.csv',
    'IU_task2_test3_split1_3.csv', 'IU_task2_val2_split6_2.csv', 'IU_task2_test3_split2_3.csv',
    'IU_task2_val2_split7_2.csv', 'IU_task2_test3_split3_3.csv', 'IU_task2_val2_split8_2.csv',
    'IU_task2_test3_split4_3.csv'
]

# Initialize rankings count dictionary
rankings_count = {
    'original': {1: 0, 2: 0, 3: 0, 4: 0},
    'Augmented 1': {1: 0, 2: 0, 3: 0, 4: 0},
    'Augmented 2': {1: 0, 2: 0, 3: 0, 4: 0},
    'Augmented 3': {1: 0, 2: 0, 3: 0, 4: 0}
}
na_count = 0 
re_count = 0
same_id = []
for index, row in index_file_df.iterrows():
    tmp_file = []
    tmp_assignment = []
    tmp_ranking = []
    id_value = row['ID']
    assignment_str = row['Assignment']
    
    assignment_mapping = {}
    for assignment in assignment_str.split(','):
        idx, name = assignment.split(' -> ')
        assignment_mapping[int(idx)] = name.strip()

    tmp_assignment.append(assignment_mapping)
    for file in other_files:
        df = pd.read_csv(file)

        
        if id_value in df['ID'].values:
            matching_row = df[df['ID'] == id_value].iloc[0]

            tmp_file.append(file)
            ranking_str = matching_row['Doctor evaluation: Ranking four reports in order from good to bad (1-4) e.g., 2,3,1,4']             
            if pd.isna(ranking_str) == False:
                if type(ranking_str) == float or type(ranking_str) == np.float64:
                    ranking_str = str(ranking_str)
                ranking_str = format_string(ranking_str)
            if pd.isna(ranking_str) == True:
                continue
            try:
                rankings = list(map(int, ranking_str.split(',')))
            except ValueError:
                continue

            tmp_ranking.append(rankings)
            for rank, report_idx in enumerate(rankings, 1):
                if report_idx not in assignment_mapping:
                    continue

                report_name = assignment_mapping[report_idx]
                rankings_count[report_name][rank] += 1
            break
    entry = {
                'id':id_value,
                'assignment':tmp_assignment,
                'ranking':tmp_ranking
            }
    same_id.append(entry)
