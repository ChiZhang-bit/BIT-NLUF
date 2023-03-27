#-*- coding:utf-8 -*-
import time

class dict_tree:
    # 构建词典树
    def __init__(self):
        self.root = {} #词典树的根节点
        self.max_length = 0 # 最大长度
        self.endtag = '[end]' # 词典树的标记，有end标记的才表示为一个词

    def insert(self, word):
        node = self.root
        for chara in word:
            node = node.setdefault(chara, {}) #添加节点
        self.max_length = max(self.max_length, len(word))
        node[self.endtag] = '[end]'

    def search(self, word):
        node = self.root
        for chara in word:
            if chara not in node:
                return 0
            node = node[chara]
        if self.endtag in node:
            return 1 #如果最后有engtag 那么这个路径是词
        else:
            return 0 #如果没有 那么在词典树中就没有找到该词

def init_dict_tree(dictfile_path):
    '''
    初始化词典树
    :param dictfile_path: 词典的路径
    :return: 形成的词典树
    '''
    dict = dict_tree()
    dictfile = open(dictfile_path, 'r' , encoding = "utf-8")
    for line in dictfile.readlines():
        dict.insert(line.strip())
    print("最长长度:{}".format(dict.max_length))
    return dict

def forward_segment(dict: dict_tree, inputdata, outputdata):
    '''
    正向匹配分词
    :param dict: 词典树
    :param inputdata: 输入文件路径
    :param outputdata: 要输出的文件路径
    :return: 返回一个列表 包含分词的结果 供双向分词使用
    '''

    inputfile = open(inputdata, "r", encoding="utf-8")
    outputfile = open(outputdata, "w", encoding="utf-8")
    result_list = []
    for line in inputfile.readlines():
        while len(line) > 0:
            max_len = dict.max_length
            if len(line) < max_len:
                max_len = len(line)
            #开始搜索
            word = line[0:max_len]
            while dict.search(word) == 0:
                if len(word) == 1:
                    break
                #没有搜索到 减少一位 继续搜索
                word = word[0:(len(word)-1)]
            if word != '\n':
                outputfile.write(word+"  ")
            else:
                outputfile.write(word)
            result_list.append(word)
            line = line[len(word):]

    inputfile.close()
    outputfile.close()

    return result_list

def backward_segment(dict: dict_tree, inputdata, outputdata):
    '''
    逆向匹配分词
    :param dict: 词典树
    :param inputdata: 输入文件路径
    :param outputdata: 输出的文件路径
    :return: 返回一个列表 包含分词的结果 供双向分词使用
    '''

    inputfile = open(inputdata, "r", encoding="utf-8")
    outputfile = open(outputdata, "w", encoding="utf-8")
    result_list = []
    for line in inputfile.readlines():
        result = []
        while len(line) > 0:
            max_len = dict.max_length
            if len(line) < max_len:
                max_len = len(line)
            #逆向查找 长度为 max_len的词语
            word = line[(len(line) - max_len):]
            while dict.search(word) == 0:
                if len(word) == 1:
                    break
                # 减少一位
                word = word[1:]
            result.append(word)
            line = line[0:(len(line) - len(word))]
        while len(result) > 0:
            res_word = result.pop()
            if res_word != '\n':
                outputfile.write(res_word+"  ")
            else:
                outputfile.write(res_word)
            result_list.append(res_word)

    inputfile.close()
    outputfile.close()

    return result_list

def count_single(wordlist : list):
    '''
    统计单字成词的个数
    :param wordlist: 分词列表
    :return: 单字成词的个数
    '''
    return  sum(1 for word in wordlist if len(word) == 1)

def bidirectional_segment(forward_list, backword_list, outputdata):
    '''
    双向匹配分词
    词数更少优先级更高
    单字更少优先级更高
    都相等时逆向匹配优先级更高
    :param forward_list: 正相匹配分词列表
    :param backword_list: 逆向匹配分词列表
    :param outputdata: 输出文件路径
    :return:
    '''

    outputfile = open(outputdata, "w", encoding="utf-8")
    result_list = []
    if len(forward_list) < len(backword_list):#词数更少优先级更高
        result_list = forward_list
        for word in forward_list:
            if word != '\n':
                outputfile.write(word+"  ")
            else:
                outputfile.write(word)
    elif len(forward_list) > len(backword_list):
        result_list = backword_list
        for word in backword_list:
            if word != '\n':
                outputfile.write(word + "  ")
            else:
                outputfile.write(word)
    else:
        if count_single(forward_list) < count_single(backword_list):#单字更少优先级更高
            result_list = forward_list
            for word in forward_list:
                if word != '\n':
                    outputfile.write(word + "  ")
                else:
                    outputfile.write(word)
        else:#都相等时逆向匹配优先级更高
            result_list = backword_list
            for word in backword_list:
                if word != '\n':
                    outputfile.write(word + "  ")
                else:
                    outputfile.write(word)
    outputfile.close()
    return result_list

def evaluate_speed(dict: dict_tree, inputdata, outputdata):
    '''
    测量分词的效率
    :param dict: 词典树
    :param inputdata: 输入文件路径
    :param outputdata: 输出文件的路径
    :return:
    '''
    start_time = time.time()
    num = 0
    with open(inputdata, 'r', encoding="utf-8") as fp:
        for line in fp.readlines():
            num = num + len(line)

    # forward_segment(dict, inputdata, outputdata)
    backward_segment(dict, inputdata, outputdata)

    elapsed_time = time.time() - start_time
    print("共{}字".format(num), end="")
    print("耗时{}s".format(elapsed_time))

def main1():
    #初始化词典和最大长度
    dict = init_dict_tree("data/dict.txt")

    #正向匹配
    forward_list = forward_segment(dict, "data/test/pku_test.utf8", "data/seg_result/forward_tree_out.txt")
    print("正向匹配分词结果为：", end="")

    '''print(forward_list)'''

    #逆向匹配
    backward_list = backward_segment(dict, "data/test/pku_test.utf8", "data/seg_result/back_tree_out.txt")
    print("逆向匹配分词结果为：", end="")
#    print(backword_list)

    #双向匹配
    bidirectional_list = bidirectional_segment(forward_list,backward_list,"data/seg_result/bidirectional_tree_out.txt")
    print("双向匹配分词结果为：", end="")

def main2():
    dict = init_dict_tree("data/dict.txt")
    evaluate_speed(dict, "data/speed/speed_test.utf8", "data/speed/2.txt")

if __name__ == "__main__":
    dict = init_dict_tree("data/dict.txt")
