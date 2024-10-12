"""
generate assist09-like data format for ednet
"""
import os
import pandas as pd

records_folder = 'D:/ZJP/EdNet-KT1/KT1'
ques_file = "D:/ZJP/EdNet-Contents/contents/questions.csv"
save_file = 'D:/ZJP/EdNet/ednet.csv'

ques_df = pd.read_csv(ques_file)

# merge all student csv in the whole
stu_file_list = os.listdir(records_folder)
path = os.path.join(records_folder, stu_file_list[0])
stu_df = pd.read_csv(path)
stu_df['user_id'] = stu_file_list[0].rstrip('.csv')

max_stu_num = 5000
count = 1
for stu_file_name in stu_file_list[1:]:
    path = os.path.join(records_folder, stu_file_name)
    tmp_df = pd.read_csv(path)
    if len(tmp_df) < 10:
        continue
    tmp_df['user_id'] = stu_file_name.rstrip('.csv')
    stu_df = pd.concat([stu_df, tmp_df])

    count += 1
    if count >= max_stu_num:
        break

df = pd.merge(stu_df, ques_df, how='inner', on=['question_id'])
df['correct'] = (df['user_answer'] == df['correct_answer'])
df['correct'] = df['correct'].apply(lambda x: int(x))
df.sort_values(by=['user_id'], ascending=True, inplace=True)
df.to_csv(save_file)
