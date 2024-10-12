from Eedi.Eedi_utils import *
import os
import pandas as pd
import numpy as np
import time


class DataProcess:
    def __init__(self, min_inter_num=10):
        self.data_path = dataPath
        self.save_folder = saveFolder
        self.min_inter_num = min_inter_num
        self.question_id_dict = None
        self.skill_id_dict = None
        # self.bundle_id_dict = None
        self.type_id_dict = None
        self.train_user_id_dict = None
        self.quesID2skillIDs_dict = {}

        self.logger = Logger()

    def process_csv(self, isRemoveEmptySkill=True, isRemoveMulSkill=False):
        print("### processing data ###")
        # read original csv file
        df = pd.read_csv(self.data_path, low_memory=False, encoding="ISO-8859-1")
        self.logger.count("originally", df)

        # 1. remove records without skill / fill empty skill
        if isRemoveEmptySkill:
            df.dropna(subset=['skill_id'], inplace=True)
            self.logger.count('After removing empty skill', df)
        else:
            df['skill_id'].fillna(value='UNK', inplace=True)

        # 2. remove or keep records with multiple-skill question
        if isRemoveMulSkill:
            df = df[~df['skill_id'].str.contains('_')]

        return df

    def get_train_test_df(self, df, train_user_ratio=0.8):
        print("\n### splitting train and test df ###")
        # get train df & test df
        all_users = list(df['user_id'].unique())
        num_train_user = int(len(all_users) * train_user_ratio)

        train_users = list(np.random.choice(all_users, size=num_train_user, replace=False))
        test_users = list(set(all_users) - set(train_users))
        open(os.path.join(self.save_folder, 'train_test', 'train_users.txt'), 'w').write(str(train_users))
        open(os.path.join(self.save_folder, 'train_test', 'test_users.txt'), 'w').write(str(test_users))
        train_df = df[df['user_id'].isin(train_users)]
        test_df = df[df['user_id'].isin(test_users)]

        # 3. remove the questions from test df, which do not exist in train df
        train_questions = list(train_df['question_id'].unique())
        test_df = test_df[test_df['question_id'].isin(train_questions)]

        # save train df & test df
        train_df.to_csv(os.path.join(self.save_folder, 'train_test', 'train_df.csv'))
        test_df.to_csv(os.path.join(self.save_folder, 'train_test', 'test_df.csv'))

        return train_df, test_df

    def encode_entity(self, train_df, test_df):
        print("\n### encoding entities ###")
        df = pd.concat([train_df, test_df], ignore_index=True)

        # encode questions
        problems = df['question_id'].unique()
        self.question_id_dict = dict(zip(problems, range(len(problems))))
        save_dict(self.question_id_dict, os.path.join(self.save_folder, "encode", "question_id_dict.txt"))
        print('question number: %d' % len(problems))

        # encode skills
        skills = df['skill_id'].astype(str).unique()
        skill_set = set(skills)
        for skill in skills:
            for s in str(skill).split('_'):
                skill_set.add(s)

        index, self.skill_id_dict = 0, dict()
        for skill in skill_set:
            if '_' not in str(skill):
                self.skill_id_dict[skill] = index
                index += 1
        for skill in skill_set:
            if skill not in self.skill_id_dict.keys():
                self.skill_id_dict[skill] = index
                index += 1
        save_dict(self.skill_id_dict, os.path.join(self.save_folder, "encode", "skill_id_dict.txt"))
        print('skill number %d' % len(skills))

        # # encode bundles
        # bundles = df['bundle_id'].unique()
        # self.bundle_id_dict = dict(zip(bundles, range(len(bundles))))
        # save_dict(self.bundle_id_dict, os.path.join(self.save_folder, "encode", "bundle_id_dict.txt"))
        # print('bundle number %d' % len(bundles))

        # encode train_users
        train_users = train_df['user_id'].unique()
        self.train_user_id_dict = dict(zip(train_users, range(len(train_users))))
        save_dict(self.train_user_id_dict, os.path.join(self.save_folder, "encode", "train_user_id_dict.txt"))
        print('train_user number: %d' % len(train_users))

    def generate_user_sequence(self, df, seq_file):
        # generate user interaction sequence
        ui_df = df.groupby(['user_id'], as_index=True)

        user_inters = []
        for ui in ui_df:
            tmp_user, tmp_inter = ui[0], ui[1]
            tmp_inter.sort_values(by=['order_id'], ascending=True, inplace=True)
            tmp_problems = list(tmp_inter['question_id'])
            tmp_skills = list(tmp_inter['skill_id'])
            tmp_ans = list(tmp_inter['correct'])
            user_inters.append([[len(tmp_inter)], tmp_skills, tmp_problems, tmp_ans])
        write_list(os.path.join(self.save_folder, "train_test", seq_file), user_inters)

    def encode_user_sequence(self, train_or_test):
        with open(os.path.join(self.save_folder, "train_test", '%s_data.txt' % train_or_test), 'r') as f:
            lines = f.readlines()

        index = 0
        seqLen_list, problems_list, skills_list, answers_list = [], [], [], []
        while index < len(lines):
            tmp_skills = eval(lines[index + 1])
            tmp_skills = [self.skill_id_dict[ele] for ele in tmp_skills]
            tmp_pro = eval(lines[index + 2])
            tmp_pro = [self.question_id_dict[ele] for ele in tmp_pro]
            tmp_ans = eval(lines[index + 3])
            real_len = len(tmp_pro)

            seqLen_list.append(real_len)
            problems_list.append(tmp_pro)
            skills_list.append(tmp_skills)
            answers_list.append(tmp_ans)

            index += 4

        with open(os.path.join(self.save_folder, "train_test", "%s_question.txt" % train_or_test), 'w') as w:
            for user in range(len(seqLen_list)):
                w.write('%d\n' % seqLen_list[user])
                w.write('%s\n' % ','.join([str(i) for i in problems_list[user]]))
                w.write('%s\n' % ','.join([str(i) for i in answers_list[user]]))

        with open(os.path.join(self.save_folder, "train_test", "%s_skill.txt" % train_or_test), 'w') as w:
            for user in range(len(seqLen_list)):
                w.write('%d\n' % seqLen_list[user])
                w.write('%s\n' % ','.join([str(i) for i in skills_list[user]]))
                w.write('%s\n' % ','.join([str(i) for i in answers_list[user]]))

        # # generate input data using bundle_id
        # get_train_test_bundle(train_or_test)

    def get_ques_skill_mat(self):
        df = pd.read_csv(os.path.join(self.save_folder, "graph", "ques_skill.csv"))
        num_ques, num_skill = df['ques'].max() + 1, df['skill'].max() + 1
        ques_skill_mat = np.zeros(shape=(num_ques, num_skill), dtype=np.int)
        for index, row in df.iterrows():
            quesID, skillID = int(row['ques']), int(row['skill'])
            ques_skill_mat[quesID][skillID] = 1
        np.save(os.path.join(self.save_folder, "graph", "ques_skill_mat.npy"), ques_skill_mat)

    def build_ques_interaction_graph(self, train_df, test_df):
        """
        build ques_skill interaction graph
        build ques_bundle interaction graph
        """
        print("building question interaction graph")
        df = pd.concat([train_df, test_df], ignore_index=True)

        ques_skill_set, ques_bundle_set, ques_type_set = set(), set(), set()
        for ques in self.question_id_dict.keys():
            quesID = self.question_id_dict[ques]
            tmp_df = df[df['question_id'] == ques]
            tmp_df_0 = tmp_df.iloc[0]

            # build ques-skill graph
            if quesID not in self.quesID2skillIDs_dict.keys():
                self.quesID2skillIDs_dict[quesID] = set()
            tmp_skills = [ele for ele in str(tmp_df_0['skill_id']).split('_')]
            for s in tmp_skills:
                skillID = self.skill_id_dict[s]
                ques_skill_set.add((quesID, skillID))
                self.quesID2skillIDs_dict[quesID].add(skillID)

            # # build ques-bundle graph
            # tmp_bundle = tmp_df_0['bundle_id']
            # ques_bundle_set.add((quesID, self.bundle_id_dict[tmp_bundle]))

        save_graph(ques_skill_set, os.path.join(self.save_folder, 'graph', 'ques_skill.csv'), ['ques', 'skill'])
        # save_graph(ques_bundle_set, os.path.join(self.save_folder, 'graph', 'ques_bundle.csv'), ['ques', 'bundle'])

        # save ques_skill matrix
        self.get_ques_skill_mat()

    def get_ques_attribute(self, train_df):
        print("\n### getting question attributes ###")
        # calculate question difficulty using train records
        quesID2diffValue_dict = get_quesDiff(train_df, self.question_id_dict)
        save_dict(quesID2diffValue_dict, os.path.join(self.save_folder, "attribute", "quesID2diffValue_dict.txt"))

        # # calculate average elapsed_time
        # quesID2AvgMsRsp_dict = get_quesAvgMsRsp(train_df, self.question_id_dict)
        # save_dict(quesID2AvgMsRsp_dict, os.path.join(self.save_folder, "attribute", "quesID2AvgMsRsp_dict.txt"))

        # categorize difficulty into 10 discrete levels
        quesID2diffLevel_df = pd.DataFrame.from_dict(quesID2diffValue_dict, orient='index', columns=['diff'])
        quesID2diffLevel_df.index.names = ['ques']
        quesID2diffLevel_df['diff'] = quesID2diffLevel_df['diff'].apply(lambda x: int(x * 10))
        quesID2diffLevel_df.to_csv(os.path.join(self.save_folder, "graph", "ques_diff.csv"))

        # # calculate question discrimination (refer to AKTHE)
        # user_score_list = []
        # for user in self.train_user_id_dict.keys():
        #     tmp_df = train_df[train_df['user_id'] == user]
        #     score = tmp_df['correct'].sum()
        #     user_score_list.append((user, score))
        # user_score_list.sort(key=lambda x: x[1], reverse=True)
        #
        # ratio, num_user = 0.5, len(user_score_list)
        # top_users, _ = zip(*user_score_list[:int(ratio * num_user)])
        # btm_users, _ = zip(*user_score_list[-int(ratio * num_user):])
        # top_df = train_df[train_df['user_id'].isin(top_users)]
        # btm_df = train_df[train_df['user_id'].isin(btm_users)]
        # quesID2topDiff = get_quesDiff(top_df, self.question_id_dict)
        # quesID2btmDiff = get_quesDiff(btm_df, self.question_id_dict)
        #
        # quesID2discValue_dict = dict()
        # for quesID in quesID2diffValue_dict.keys():
        #     disc = quesID2topDiff[quesID] - quesID2btmDiff[quesID]
        #     quesID2discValue_dict[quesID] = disc
        # # quesID_disc_list = sorted(quesID2discValue_dict.items(), key=lambda x: x[1], reverse=True)
        # # print(quesID_disc_list)
        # save_dict(quesID2discValue_dict, os.path.join(self.save_folder, "attribute", "quesID2discValue_dict.txt"))

        # # categorize discrimination into 4 discrete levels
        # with open(os.path.join(self.save_folder, "graph", "ques_disc.csv"), 'w', encoding='utf-8') as writer:
        #     writer.write("ques,disc\n")
        #     for quesID, discValue in quesID2discValue_dict.items():
        #         if discValue < 0.2:
        #             discLevel = 0
        #         elif 0.2 <= discValue < 0.3:
        #             discLevel = 1
        #         elif 0.3 <= discValue < 0.4:
        #             discLevel = 2
        #         else:
        #             discLevel = 3
        #         writer.write("%d,%d\n" % (quesID, discLevel))

    def build_stu_interaction_graph(self, train_df):
        """
        build stu_skill interaction graph
        build stu_question interaction graph
        """
        print("\n### building student interaction graph ###")
        df = train_df.copy()
        # df['elapsed_time'] = feature_normalize(df['elapsed_time'])

        stu_skill_set, stu_ques_set = set(), set()
        num_train_stu, num_skill = len(self.train_user_id_dict), len(
            [s for s in self.skill_id_dict.keys() if '_' not in str(s)])
        stu_skill_mat = np.zeros(shape=(num_train_stu, num_skill), dtype=np.float32)
        for stu in self.train_user_id_dict.keys():  # traverse all students in train dataset
            stuID = self.train_user_id_dict[stu]
            tmp_df = df[df['user_id'] == stu].copy()
            tmp_df.sort_values(by=['order_id'], ascending=True, inplace=True)

            # # build stu-skill graph, using combined skills
            # for skill in tmp_df['skill_id'].unique():
            #     skillID = self.skill_id_dict[skill]
            #     skill_df = tmp_df[tmp_df['skill_id'] == skill]
            #     crtRatio = skill_df['correct'].mean()
            #     wrgRatio = 1 - crtRatio
            #     stu_skill_mat[stuID][skillID] = crtRatio - wrgRatio
            #     stu_skill_set.add((stuID, skillID, crtRatio))

            # build stu_skill graph, using atom skills
            skill2crts_dict = dict()
            for index, row in tmp_df.iterrows():
                quesID = self.question_id_dict[row['question_id']]
                for skillID in self.quesID2skillIDs_dict[quesID]:
                    if skillID not in skill2crts_dict.keys():
                        skill2crts_dict[skillID] = []
                    skill2crts_dict[skillID].append(int(row['correct']))
            for skillID, correct_list in skill2crts_dict.items():
                crtRatio = np.mean(correct_list)
                wrgRatio = 1 - crtRatio
                stu_skill_mat[stuID][skillID] = crtRatio - wrgRatio
                stu_skill_set.add((stuID, skillID, crtRatio))

            # build stu_ques graph
            timeStep = 1
            for index, row in tmp_df.iterrows():
                quesID = self.question_id_dict[row['question_id']]
                correct = row['correct']
                timePoint = timeStep / len(tmp_df)
                # elapsed_time = row['elapsed_time']
                stu_ques_set.add((stuID, quesID, correct, timePoint))
                timeStep += 1

        names = ['stu', 'skill', 'crtRatio']
        save_graph(stu_skill_set, os.path.join(self.save_folder, "graph", "stu_skill.csv"), names)
        names = ['stu', 'ques', 'correct', 'timePoint']
        save_graph(stu_ques_set, os.path.join(self.save_folder, "graph", "stu_ques.csv"), names)
        np.save(os.path.join(self.save_folder, "graph", "stu_skill_mat.npy"), stu_skill_mat)

        # cluster students and skills according to stu_skill_mat
        for num_cluster in [8, 16, 32, 60, 120]:
            get_cluster(num_cluster, stu_skill_mat,
                        os.path.join(self.save_folder, "graph", "stu_cluster_%d.csv" % num_cluster),
                        ['stu', 'cluster'])

        skill_stu_mat = np.transpose(stu_skill_mat)
        for num_cluster in [8, 16, 32, 60, 120]:
            get_cluster(num_cluster, skill_stu_mat,
                        os.path.join(self.save_folder, "graph", "skill_cluster_%d.csv" % num_cluster),
                        ['skill', 'cluster'])


if __name__ == '__main__':
    t = time.time()
    dataPath = "D:/ZJP/Eedi/Eedi.csv"
    saveFolder = './'

    DP = DataProcess()
    DF = DP.process_csv()
    # trainDF, testDF = DP.get_train_test_df(DF)
    # DP.encode_entity(trainDF, testDF)
    #
    # DP.build_ques_interaction_graph(trainDF, testDF)
    # DP.get_ques_attribute(trainDF)
    # DP.build_stu_interaction_graph(trainDF)
    #
    # DP.generate_user_sequence(trainDF, 'train_data.txt')
    # DP.generate_user_sequence(testDF, 'test_data.txt')
    # DP.encode_user_sequence(train_or_test='train')
    # DP.encode_user_sequence(train_or_test='test')
    #
    # print("consume %d seconds" % (time.time() - t))
