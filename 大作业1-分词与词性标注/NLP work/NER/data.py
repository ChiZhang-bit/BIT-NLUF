'''
用作读取数据
'''
from os.path import join
from codecs import open


def build_corpus(filepath):
    """读取数据"""

    word_lists = [] # 词列表
    tag_lists = [] # 状态列表
    with open(filepath, 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

        return word_lists, tag_lists
