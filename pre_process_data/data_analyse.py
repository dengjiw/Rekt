import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# data_path = "D:/ZJP/ASSIST12/2012-2013-data-with-predictions-4-final.csv"
data_path = "./train_test/train_df.csv"
df = pd.read_csv(data_path, encoding="ISO-8859-1")
# delete the users whose interaction number is less than min_inter_num
users = df.groupby(['user_id'], as_index=True)
delete_users = []
for u in users:
    if len(u[1]) < 10:
        delete_users.append(u[0])
print('deleted user number based min-inters %d' % len(delete_users))
df = df[~df['user_id'].isin(delete_users)]
print('After deleting some users, records number %d' % len(df))
# select_questions = list(df['problem_id'].value_counts().index)[:20000]
# delete_questions = list(df['problem_id'].value_counts().index)[20000:]
# del_df = df[df['problem_id'].isin(delete_questions)]
# del_users = set(del_df['user_id'])
# df = df[~df['user_id'].isin(del_users)]

select_schools = list(df['school_id'].value_counts().index)[:20]
df = df[df['school_id'].isin(select_schools)]
# select_schools = list(df['student_class_id'].value_counts().index)[:100]
# df = df[df['student_class_id'].isin(select_schools)]
# select_students = list(df['user_id'].value_counts().index)[:5000]
select_students = np.random.choice(np.array(df['user_id'].unique()), size=5000, replace=False)
df = df[df['user_id'].isin(select_students)]

# df['school_id'], uniques_sch = df['school_id'].factorize()
# df['problem_id'], uniques_ques = df['problem_id'].factorize()
# df['student_class_id'], uniques_class = df['student_class_id'].factorize()
# num_school = len(set(df['school_id']))
# num_ques = len(set(df['problem_id']))
# num_class = len(set(df['student_class_id']))

print("num of records", len(df))
print("num of students", len(set(df['user_id'])))
print("num of questions", len(set(df['problem_id'])))

# stu_ques_mat = np.zeros(shape=(num_class, num_ques), dtype=np.int8)
# for index, row in df.iterrows():
#     schID, quesID = int(row['student_class_id']), int(row['problem_id'])
#     stu_ques_mat[schID][quesID] = 1
#
# sns.heatmap(stu_ques_mat)
# plt.show()
