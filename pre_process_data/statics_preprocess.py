"""
generate assist09-like data format for statics2011
"""
import numpy as np
import pandas as pd


def get_kc(series):
    KC = series['KC (F2011)']
    if not pd.isna(series['KC (F2011).1']):
        KC += ';' + series['KC (F2011).1']
    if not pd.isna(series['KC (F2011).2']):
        KC += ';' + series['KC (F2011).2']
    return KC


if __name__ == '__main__':
    logs_file = 'D:/ZJP/static2011/ds507_tx_2020_1004_215826/ds507_tx_All_Data_1664_2017_0227_034415.txt'
    save_file = 'D:/ZJP/static2011/static2011.csv'
    df = pd.read_csv(logs_file, low_memory=False, delimiter='\t')
    df.sort_values(by=['Time', 'Anon Student Id'], ascending=True, inplace=True)

    # 只保留每个（题目，步骤）的第一次做题记录
    print('Before remove multiple attempts of the same step: ', len(df))
    df.drop_duplicates(subset=['Anon Student Id', 'Session Id', 'Problem Name', 'Step Name'], keep='first', inplace=True)
    print('After remove multiple attempts of the same step: ', len(df))

    df.dropna(subset=['Outcome'], inplace=True)
    df['Outcome'] = df['Outcome'].map({'CORRECT': 1, 'INCORRECT': 0, 'HINT': 0})

    df["KC"] = df.apply(get_kc, axis=1)
    df['Problem_Step Name'] = df['Problem Name'].map(str) + '&' + df['Step Name'].map(str)

    df['Duration (sec)'] = df['Duration (sec)'].map({'.': np.nan})
    df['Duration (sec)'] = df['Duration (sec)'].fillna(method="ffill")

    old_names = ['Time', 'Anon Student Id', 'Problem_Step Name', 'Problem Name', 'KC', 'Outcome', 'Duration (sec)']
    new_names = ['order_id', 'user_id', 'question_id', 'bundle_id', 'skill_id', 'correct', 'elapsed_time']
    df.rename(columns={key: value for key, value in zip(old_names, new_names)}, inplace=True)
    df[new_names].to_csv(save_file, sep=',')
