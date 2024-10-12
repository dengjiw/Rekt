"""
Generate ASSIST09-like format for ASSIST17
"""
import numpy as np
import pandas as pd


if __name__ == '__main__':
    logs_file = 'D:/ZJP/ASSIST17/anonymized_full_release_competition_dataset.csv'
    save_file = 'D:/ZJP/ASSIST17/ASSIST17.csv'

    old_names = ['studentId', 'skill', 'problemId', 'problemType', 'assistmentId', 'startTime', 'endTime', 'timeTaken',
                 'correct', 'original', 'scaffold', 'attemptCount']
    df = pd.read_csv(logs_file, low_memory=False, delimiter=',', usecols=old_names)
    df.sort_values(by=['startTime'], ascending=True, inplace=True)
    df['order_id'] = range(len(df))

    new_names = ['user_id', 'skill_id', 'question_id', 'question_type', 'assistment_id', 'startTime', 'endTime', 'duration',
                 'correct', 'original', 'scaffold', 'attempt_count']
    df.rename(columns={key: value for key, value in zip(old_names, new_names)}, inplace=True)
    df.to_csv(save_file, sep=',', index=False)
