import numpy as np

def get_tag(postagstr):
    '''
    用于提取词性
    :param postagstr:已经标注过词性的一行字符串
    :return: 提取出每一个词的词性，以列表形式返回
    '''
    postaglist = postagstr.strip().split(" ")
    res = []

    for word in postaglist:
        wordlist = word.split("/")
        pretag = wordlist[-1]
        if ']' in wordlist[-1]:
            pretaglist = wordlist[-1].split("]")
            pretag = pretaglist[0]
        res.append(pretag)
    # print(res)
    return res

def cal_postag(testfile, goldfile):
    '''
    用于测量正确值
    :param testfile: 测试集的文件路径
    :param goldfile: 答案集的文件路径
    :return: 返回准确值
    '''
    # calculate f-score 计算f-score
    f_gold = open(goldfile, "r", encoding='UTF-8')
    f_test = open(testfile, "r", encoding='UTF-8')
    gold = f_gold.readlines()
    test = f_test.readlines()

    total_correct = 0
    total_word = 0

    for gold_line, test_line in zip(gold, test):
        gold_tag = get_tag(gold_line)
        test_tag = get_tag(test_line)

        for res_gold, res_test in zip(gold_tag, test_tag):
            # print("{}=={}".format(res_gold,res_test))
            if res_gold == res_test:
                total_correct += 1
            total_word += 1

    # print(total_word)
    # print(total_correct)
    print(total_correct/total_word)

if __name__ == '__main__':
    cal_postag("data/postag/pos_ans.utf8","data/postag/pos_test_gold.utf8")
