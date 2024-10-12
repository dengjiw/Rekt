import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans


class Logger:
    def __init__(self):
        self.writer = open("./log.txt", 'a', encoding='utf-8')

    def count(self, say, df):
        log = "%s, record num: %d, student num：%d, question num：%d, skill num: %d" % (
            say, len(df), len(df['user_id'].unique()), len(df['question_id'].unique()), len(df['skill_id'].unique()))
        self.writer.write(log + '\n')
        print(log)


def save_dict(dict_to_save, file_path):
    with open(file_path, 'w') as f:
        f.write(str(dict_to_save))


def save_list(list_to_save, file_path):
    with open(file_path, 'w') as f:
        f.write(str(list_to_save))


def save_graph(tuple_set, file_path, names):
    with open(file_path, 'w', encoding='utf-8') as writer:
        writer.write("%s\n" % ','.join(names))
        for tp in tuple_set:
            writer.write("%s\n" % ','.join([str(e) for e in tp]))


def load_dict(file_path):
    with open(file_path, 'r') as f:
        dict_load = eval(f.read())
    return dict_load


def write_list(file, data):
    with open(file, 'w') as f:
        for dd in data:
            for d in dd:
                f.write(str(d) + '\n')


def feature_normalize(array):
    mu = np.mean(array, axis=0)
    sigma = np.std(array, axis=0)
    return (array - mu) / sigma


def get_quesDiff(train_df, question_id_dict):
    print(train_df['question_id'].value_counts().values)
    ques_few_cnt, quesID2diffValue_dict = 0, dict()
    for ques in question_id_dict.keys():
        tmp_df = train_df[train_df['question_id'] == ques]
        if len(tmp_df) >= 3:
            crt_ratio = tmp_df['correct'].mean()
            diff = crt_ratio
        else:
            diff = 0.5
            ques_few_cnt += 1
        quesID = question_id_dict[ques]
        quesID2diffValue_dict[quesID] = diff
    print("%d/%d questions has few attempts in train set" % (ques_few_cnt, len(question_id_dict)))
    return quesID2diffValue_dict


def get_quesDuration(train_df, question_id_dict):
    quesID2Duration_dict = dict()
    for ques in question_id_dict.keys():
        tmp_df = train_df[train_df['question_id'] == ques]
        ms = tmp_df['duration'].abs().mean()
        quesID = question_id_dict[ques]
        quesID2Duration_dict[quesID] = ms
    return quesID2Duration_dict


def get_cluster(num_cluster, data, savePath, names):
    k_means = KMeans(n_clusters=num_cluster, random_state=0).fit(data)
    clusters = list(k_means.labels_)
    print(k_means.n_iter_, clusters)
    x_cluster_set = set()
    for x_id, c_id in enumerate(clusters):
        x_cluster_set.add((x_id, c_id))
    save_graph(x_cluster_set, savePath, names)


def get_train_test_template(train_or_test):
    ques_tmp_df = pd.read_csv("graph/ques_template.csv")
    quesID2tmpID_dict = dict(zip(list(ques_tmp_df['ques']), list(ques_tmp_df['template'])))

    with open('train_test/%s_question.txt' % train_or_test, 'r') as f:
        lines = f.readlines()
        
    index = 0
    seqLen_list, questions_list, answers_list = [], [], []
    while index < len(lines):
        seqLen = eval(lines[index])
        questions = [eval(ele) for ele in lines[index + 1].split(',')]
        answers = [eval(ele) for ele in lines[index + 2].split(',')]

        seqLen_list.append(seqLen)
        questions_list.append(questions)
        answers_list.append(answers)

        index += 3

    with open('train_test/%s_template.txt' % train_or_test, 'w', encoding='utf-8') as w:
        for user in range(len(seqLen_list)):
            w.write('%d\n' % seqLen_list[user])
            w.write('%s\n' % ','.join([str(quesID2tmpID_dict[i]) for i in questions_list[user]]))
            w.write('%s\n' % ','.join([str(i) for i in answers_list[user]]))


# -----------------------------------------------------
# get student cluster
# stu_skill_mat = np.load("graph/stu_skill_mat.npy")
# for n_cluster in [140, 160]:
#     get_cluster(n_cluster, stu_skill_mat, "graph/stu_cluster_%d.csv" % n_cluster, ['stu', 'cluster'])
