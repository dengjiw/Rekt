"""
generate ASSIST09-like data format for Eedi
"""
import numpy as np
import pandas as pd

train_file = 'D:/ZJP/Eedi/data/data/train_data/train_task_1_2.csv'
test_file = 'D:/ZJP/Eedi/data/data/test_data/test_private_answers_task_1.csv'
ques_file = 'question_metadata.csv'
answer_file = 'D:/ZJP/Eedi/data/data/metadata/answer_metadata_task_1_2.csv'
save_file = 'D:/ZJP/Eedi/Eedi.csv'

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
ques_df = pd.read_csv(ques_file)
answer_df = pd.read_csv(answer_file)

data_df = pd.concat([train_df, test_df], join="inner")
print(data_df.head())
data_df = pd.merge(data_df, ques_df, how='inner', on=['QuestionId'])
data_df = pd.merge(data_df, answer_df, how='inner', on=['AnswerId'])
print(data_df.columns)

# delete the users whose interaction number is less than min_inter_num
min_iter_num = 10
delete_users = []
users = data_df.groupby(['UserId'], as_index=True)
for u in users:
    if len(u[1]) < min_iter_num:
        delete_users.append(u[0])
print('deleted user number based min-inters %d' % len(delete_users))
data_df = data_df[~data_df['UserId'].isin(delete_users)]

# select 5000 users randomly
max_num_stu = 5000
select_users = np.random.choice(np.array(data_df['UserId'].unique()), size=max_num_stu, replace=False)
data_df = data_df[data_df['UserId'].isin(select_users)]
print("select users done")

# data_df['SubjectId'] = data_df['SubjectId'].map(lambda x: '_'.join([str(e) for e in eval(x)]))

old_names = ['DateAnswered', 'UserId', 'QuestionId', 'SubjectId', 'IsCorrect']
new_names = ['order_id', 'user_id', 'question_id', 'skill_id', 'correct']
data_df.rename(columns={key: value for key, value in zip(old_names, new_names)}, inplace=True)

for col in new_names:
    print("column %s number: %d" % (col, len(set(data_df[col]))))

data_df[new_names].to_csv(save_file)
