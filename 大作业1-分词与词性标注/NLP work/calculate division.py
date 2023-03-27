import numpy as np


def cal_fscore(testfile, goldfile):
    '''

    :param testfile: 测试集的文件路径
    :param goldfile: 答案集的文件路径
    :return: 返回三个值 P R F1
    '''
    # calculate f-score 计算f-score

    f_gold = open(goldfile, "r", encoding='UTF-8')
    f_test = open(testfile, "r", encoding='UTF-8')

    total_gold_seg = 0
    for line in f_gold.readlines():
        total_gold_seg += len(line.split())

    total_test_seg = 0
    for line in f_test.readlines():
        total_test_seg += len(line.split())

    total_correct_seg = 0 #所有分词正确的数量
    total_error_seg = 0 #所有错误分词的数量

    f_gold.seek(0)
    f_test.seek(0)
    for line_gold, line_test in zip(f_gold.readlines(), f_test.readlines()):
        dict_gold = {}
        dict_test = {}
        for words in line_gold.strip().split():
            if words in dict_gold:
                dict_gold[words] += 1
            else:
                dict_gold[words] = 1
        for words in line_test.strip().split():
            if words in dict_test:
                dict_test[words] += 1
            else:
                dict_test[words] = 1

        for words in dict_test:
            if words in dict_gold:
                if dict_gold[words] >= dict_test[words]:
                    total_correct_seg += dict_test[words]
                else:
                    total_correct_seg += dict_gold[words]
                    total_error_seg += dict_test[words] - dict_gold[words]
            else:
                total_error_seg += dict_test[words]


    P_score = total_correct_seg / total_test_seg
    R_score = total_correct_seg / total_gold_seg
    # F_score = 2 * total_correct_seg / (total_gold_seg + total_test_seg)
    F_score = 2 * P_score * R_score / (P_score + R_score)
    print("Precious = ", P_score)
    print("Recall = ", R_score)
    print("F-Score = ", F_score)
    f_test.close()
    f_gold.close()



if __name__ == '__main__':
    cal_fscore("data\\seg_result\\forward_tree_out.txt","data\pku_test_gold.utf8")