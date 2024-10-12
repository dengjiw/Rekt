from ASSIST17.assist17_utils import *
import os
import pandas as pd
import numpy as np
import time


class DataProcess:
    def __init__(self):
        self.data_path = dataPath
        self.save_folder = saveFolder
        self.question_id_dict = None
        self.skill_id_dict = None
        # self.question_id_dict = eval(open("encode/question_id_dict.txt").read())
        # self.skill_id_dict = eval(open("encode/skill_id_dict.txt").read())
        self.template_id_dict = None
        self.type_id_dict = None
        self.train_user_id_dict = None
        self.quesID2skillIDs_dict = {}

        self.logger = Logger()

    def process_csv(self, isRemoveEmptySkill=False, isRemoveMulSkill=False, isRemoveScaffold=True, min_iter_num=3):
        print("### processing data ###")
        # read original csv file
        df = pd.read_csv(self.data_path, encoding="ISO-8859-1")
        self.logger.count("originally", df)

        # 1. remove duplicated rows by 'order_id'

        # 2. sort records by 'order_id' in ascending order
        df.sort_values(by=['order_id', 'user_id'], ascending=True, inplace=True)

        # 3. remove records without skill or fill empty skill
        if isRemoveEmptySkill:
            df.dropna(subset=['skill_id'], inplace=True)
            df = df[~df['skill_id'].isin(['noskill'])]
            self.logger.count('After removing empty skill', df)
        else:
            df['skill_id'].fillna(value='UNK', inplace=True)
            print('empty skill have been filled')

        # 4. remove or keep records with multiple-skill question
        # if isRemoveMulSkill:
        #     df = df[~df['skill_id'].str.contains(skill_spliter)]
        #     self.logger.count("After removing multiple-skill", df)

        # 5. remove or keep scaffolding questions
        if isRemoveScaffold:
            df = df[df['original'].isin([1])]
            self.logger.count('After removing scaffolding problems', df)

        # 6. delete the users whose interaction number is less than min_inter_num
        users = df.groupby(['user_id'], as_index=True)
        delete_users = []
        for u in users:
            if len(u[1]) < min_iter_num:
                delete_users.append(u[0])
        print('deleted user number based min-inters %d' % len(delete_users))
        df = df[~df['user_id'].isin(delete_users)]
        self.logger.count('After deleting some users', df)

        return df

    def split_train_test_df(self, df, train_user_ratio=0.8):
        print("\n### splitting train and test df ###")
        # get train df & test df
        all_users = list(df['user_id'].unique())
        num_train_user = int(len(all_users) * train_user_ratio)

        train_users = list(np.random.choice(all_users, size=num_train_user, replace=False))
        test_users = list(set(all_users) - set(train_users))
        save_list(train_users, os.path.join(self.save_folder, 'train_test', 'train_users.txt'))
        save_list(test_users, os.path.join(self.save_folder, 'train_test', 'test_users.txt'))
        train_df = df[df['user_id'].isin(train_users)]
        test_df = df[df['user_id'].isin(test_users)]

        # 8. remove the questions that do not exist in train df
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
        questions = df['question_id'].unique()
        self.question_id_dict = dict(zip(questions, range(len(questions))))
        save_dict(self.question_id_dict, os.path.join(self.save_folder, "encode", "question_id_dict.txt"))
        print('question number: %d' % len(questions))

        # encode skills
        skills = df['skill_id'].astype(str).unique()
        skill_set = set(skills)

        # for skill in skills:
        #     for s in str(skill).split(skill_spliter):
        #         skill_set.add(s)

        index, self.skill_id_dict = 0, dict()

        # for skill in skill_set:
        #     if skill_spliter not in str(skill):
        #         self.skill_id_dict[skill] = index
        #         index += 1

        for skill in skill_set:
            if skill not in self.skill_id_dict.keys():
                self.skill_id_dict[skill] = index
                index += 1

        save_dict(self.skill_id_dict, os.path.join(self.save_folder, "encode", "skill_id_dict.txt"))
        print("skill_id_dict", len(self.skill_id_dict))
        print('skill number: %d' % len(skills))

        # # encode templates
        # templates = df['template_id'].unique()
        # self.template_id_dict = dict(zip(templates, range(len(templates))))
        # save_dict(self.template_id_dict, os.path.join(self.save_folder, "encode", "template_id_dict.txt"))
        # print('template number: %d' % len(templates))

        # encode question_type
        types = df['question_type'].unique()
        self.type_id_dict = dict(zip(types, range(len(types))))
        save_dict(self.type_id_dict, os.path.join(self.save_folder, "encode", "type_id_dict.txt"))
        print('question type: ', self.type_id_dict)

        # encode train_users
        train_users = train_df['user_id'].unique()
        self.train_user_id_dict = dict(zip(train_users, range(len(train_users))))
        save_dict(self.train_user_id_dict, os.path.join(self.save_folder, "encode", "train_user_id_dict.txt"))
        print('train_user number: %d' % len(train_users))

    def build_ques_interaction_graph(self, train_df, test_df):
        """
        build ques_skill interaction graph
        build ques_template interaction graph
        """
        df = pd.concat([train_df, test_df], ignore_index=True)

        ques_skill_set, ques_template_set, ques_type_set = set(), set(), set()
        for ques in self.question_id_dict.keys():
            quesID = self.question_id_dict[ques]
            tmp_df = df[df['question_id'] == ques]
            tmp_df_0 = tmp_df.iloc[0]

            # build ques-skill graph
            if quesID not in self.quesID2skillIDs_dict.keys():
                self.quesID2skillIDs_dict[quesID] = set()
            tmp_skills = [ele for ele in str(tmp_df_0['skill_id']).split(skill_spliter)]
            for s in tmp_skills:
                skillID = self.skill_id_dict[s]
                ques_skill_set.add((quesID, skillID))
                self.quesID2skillIDs_dict[quesID].add(skillID)

            # # build ques-template graph
            # tmp_template = tmp_df_0['template_id']
            # ques_template_set.add((quesID, self.template_id_dict[tmp_template]))

            # build ques-type graph
            tmp_type = tmp_df_0['question_type']
            ques_type_set.add((quesID, self.type_id_dict[tmp_type]))

        save_graph(ques_skill_set, os.path.join(self.save_folder, 'graph', 'ques_skill.csv'), ['ques', 'skill'])
        # save_graph(ques_template_set, os.path.join(self.save_folder, 'graph', 'ques_template.csv'), ['ques', 'template'])
        save_graph(ques_type_set, os.path.join(self.save_folder, 'graph', 'ques_type.csv'), ['ques', 'type'])

        # save ques_skill matrix
        df = pd.read_csv(os.path.join(self.save_folder, "graph", "ques_skill.csv"))
        num_ques, num_skill = df['ques'].max() + 1, df['skill'].max() + 1
        print("get_ques_skill_mat: num_ques=%d, num_skill=%d" % (num_ques, num_skill))
        ques_skill_mat = np.zeros(shape=(num_ques, num_skill), dtype=np.int)
        for index, row in df.iterrows():
            quesID, skillID = int(row['ques']), int(row['skill'])
            ques_skill_mat[quesID][skillID] = 1
        np.save(os.path.join(self.save_folder, "graph", "ques_skill_mat.npy"), ques_skill_mat)

    def get_ques_attribute(self, train_df):
        print("\n### getting question attributes ###")
        # calculate question difficulty using train records
        quesID2diffValue_dict = get_quesDiff(train_df, self.question_id_dict)
        save_dict(quesID2diffValue_dict, os.path.join(self.save_folder, "attribute", "quesID2diffValue_dict.txt"))

        # calculate average duration
        quesID2Duration_dict = get_quesDuration(train_df, self.question_id_dict)
        save_dict(quesID2Duration_dict, os.path.join(self.save_folder, "attribute", "quesID2Duration_dict.txt"))

    def build_stu_interaction_graph(self, train_df):
        """
        build stu_skill interaction graph
        build stu_question interaction graph
        """
        print("\n### building student interaction graph ###")
        df = train_df.copy()
        df['attempt_count'] = feature_normalize(df['attempt_count'])
        df['duration'] = feature_normalize(df['duration'])

        stu_skill_set, stu_ques_set = set(), set()
        num_train_stu, num_skill = len(self.train_user_id_dict), len(
            [s for s in self.skill_id_dict.keys() if skill_spliter not in str(s)])
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
                attempt_count = row['attempt_count']
                duration = row['duration']
                stu_ques_set.add((stuID, quesID, correct, timePoint, attempt_count, duration))
                timeStep += 1

        names = ['stu', 'skill', 'crtRatio']
        save_graph(stu_skill_set, os.path.join(self.save_folder, "graph", "stu_skill.csv"), names)
        names = ['stu', 'ques', 'correct', 'timePoint', 'attempt_count', 'duration']
        save_graph(stu_ques_set, os.path.join(self.save_folder, "graph", "stu_ques.csv"), names)
        np.save(os.path.join(self.save_folder, "graph", "stu_skill_mat.npy"), stu_skill_mat)

        # cluster students and skills according to stu_skill_mat
        for num_cluster in [60, 80, 100, 120]:
            get_cluster(num_cluster, stu_skill_mat,
                        os.path.join(self.save_folder, "graph", "stu_cluster_%d.csv" % num_cluster),
                        ['stu', 'cluster'])

        skill_stu_mat = np.transpose(stu_skill_mat)
        for num_cluster in [20, 40, 60, 80]:
            get_cluster(num_cluster, skill_stu_mat,
                        os.path.join(self.save_folder, "graph", "skill_cluster_%d.csv" % num_cluster),
                        ['skill', 'cluster'])
            
    def generate_user_sequence(self, df, seq_file):
        # generate user interaction sequence
        ui_df = df.groupby(['user_id'], as_index=True)

        user_inters = []
        for ui in ui_df:
            tmp_user, tmp_inter = ui[0], ui[1]
            tmp_inter.sort_values(by=['order_id'], ascending=True, inplace=True)
            tmp_questions = list(tmp_inter['question_id'])
            tmp_skills = list(tmp_inter['skill_id'])
            tmp_ans = list(tmp_inter['correct'])
            user_inters.append([[len(tmp_inter)], tmp_skills, tmp_questions, tmp_ans])
        write_list(os.path.join(self.save_folder, "train_test", seq_file), user_inters)

    def encode_user_sequence(self, train_or_test):
        with open(os.path.join(self.save_folder, "train_test", '%s_data.txt' % train_or_test), 'r') as f:
            lines = f.readlines()

        index = 0
        seqLen_list, questions_list, skills_list, answers_list = [], [], [], []
        while index < len(lines):
            tmp_skills = eval(lines[index + 1])
            tmp_skills = [self.skill_id_dict[str(ele)] for ele in tmp_skills]
            tmp_pro = eval(lines[index + 2])
            tmp_pro = [self.question_id_dict[ele] for ele in tmp_pro]
            tmp_ans = eval(lines[index + 3])
            real_len = len(tmp_pro)

            seqLen_list.append(real_len)
            questions_list.append(tmp_pro)
            skills_list.append(tmp_skills)
            answers_list.append(tmp_ans)

            index += 4

        with open(os.path.join(self.save_folder, "train_test", "%s_question.txt" % train_or_test), 'w') as w:
            for user in range(len(seqLen_list)):
                w.write('%d\n' % seqLen_list[user])
                w.write('%s\n' % ','.join([str(i) for i in questions_list[user]]))
                w.write('%s\n' % ','.join([str(i) for i in answers_list[user]]))

        with open(os.path.join(self.save_folder, "train_test", "%s_skill.txt" % train_or_test), 'w') as w:
            for user in range(len(seqLen_list)):
                w.write('%d\n' % seqLen_list[user])
                w.write('%s\n' % ','.join([str(i) for i in skills_list[user]]))
                w.write('%s\n' % ','.join([str(i) for i in answers_list[user]]))

        # # generate input data using template_id
        # get_train_test_template(train_or_test)


if __name__ == '__main__':
    t = time.time()
    dataPath = "D:/ZJP/ASSIST17/ASSIST17.csv"
    saveFolder = './'
    skill_spliter = '###'

    DP = DataProcess()
    DF = DP.process_csv()
    trainDF, testDF = DP.split_train_test_df(DF)
    DP.encode_entity(trainDF, testDF)

    DP.build_ques_interaction_graph(trainDF, testDF)
    DP.get_ques_attribute(trainDF)
    DP.build_stu_interaction_graph(trainDF)

    DP.generate_user_sequence(trainDF, 'train_data.txt')
    DP.generate_user_sequence(testDF, 'test_data.txt')
    DP.encode_user_sequence(train_or_test='train')
    DP.encode_user_sequence(train_or_test='test')

    print("consume %d seconds" % (time.time() - t))
