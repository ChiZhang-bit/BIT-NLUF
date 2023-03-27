#-*- coding:utf-8 -*-
import sys
import re
import time

class HMMSegment:
    def __init__(self):
        self.epsilon = sys.float_info.epsilon
        self.state_list = ['B', 'M', 'E', 'S']
        self.start_p = {} # 初始化概率矩阵
        self.trans_p = {} # 状态转移矩阵
        self.emit_p = {} # 发射矩阵
        self.state_dict = {} #状态集合
        self.__init_parameters()

    def __init_parameters(self):
        for state in self.state_list:
            self.start_p[state] = 1/len(self.state_list)
            self.trans_p[state] = {s: 1/len(self.state_list) for s in self.state_list}
            self.emit_p[state] = {}
            self.state_dict[state] = 0

    def __label(self, word):
        out = []
        if len(word) == 1:
            out = ['S']
        else:
            out += ['B'] + ['M'] * (len(word) - 2) + ['E']
        return out

    def train(self, dataset: list):
        '''
        训练，包含三个矩阵初始矩阵 发射矩阵 状态矩阵的填充
        :param dataset: 数据集
        :return:
        '''
        if not dataset or len(dataset) == 0:
            print('数据为空')
            return

        line_nb = 0
        for line in dataset:
            line = line.strip()
            if not line:
                continue
            line_nb += 1

            char_list = [c for c in line if c != ' ']
            word_list = line.split()
            state_list = []
            for word in word_list:
                state_list.extend(self.__label(word))

            assert len(state_list) == len(char_list)

            for index, state in enumerate(state_list):
                self.state_dict[state] += 1

                if index == 0:
                    self.start_p[state] += 1
                else:
                    self.trans_p[state_list[index-1]][state] += 1
                self.emit_p[state_list[index]][char_list[index]] \
                    = self.emit_p[state_list[index]].get(char_list[index], 0) + 1

        # 训练得到 初始概率矩阵
        self.start_p = {state: (num+self.epsilon)/line_nb for state, num in self.start_p.items()}

        # 训练得到 转移概率矩阵
        self.trans_p = {
            pre_state:
                {
                    cur_state: (cur_num+self.epsilon)/self.state_dict[pre_state] for cur_state, cur_num in value.items()
                } for pre_state, value in self.trans_p.items()
        }

        # 训练得到 发射概率矩阵
        self.emit_p = {
            state:
                {
                    char: (char_num+self.epsilon)/self.state_dict[state] for char, char_num in value.items()
                } for state, value in self.emit_p.items()
        }
        print('训练完成')

    def __viterbi(self, sentence: str):
        '''

        :param sentence:需要划分的句子
        :return: 最优最大概率的状态序列
        '''
        dp = [{}]
        path = {}
        for state in self.state_list:
            dp[0][state] = self.start_p[state]*self.emit_p[state].get(sentence[0], self.epsilon)
            path[state] = [state]

        for index in range(1, len(sentence)):
            dp.append({})
            new_path = {}

            for cur_state in self.state_list:
                emitp = self.emit_p[cur_state].get(sentence[index], self.epsilon)
                (prob, pre_state) = max(
                    [(dp[index - 1][pre_state] * self.trans_p[pre_state].get(cur_state, self.epsilon) * emitp, pre_state)
                     for pre_state in self.state_list if dp[index - 1][pre_state] > 0]
                )
                dp[index][cur_state] = prob
                new_path[cur_state] = path[pre_state]+[cur_state]
            path = new_path

        if self.emit_p['M'].get(sentence[-1], self.epsilon) > \
                self.emit_p['S'].get(sentence[-1], self.epsilon):
            (prob, state) = max([(dp[len(sentence)-1][state], state) for state in ('E', 'M')])
        else:
            (prob, state) = max([(dp[len(sentence)-1][state], state) for state in self.state_list])

        return prob, path[state]

    def cut(self, sentence: str):
        '''
        切分句子 进行分词
        :param sentence:句子
        :return: 分词的序列
        '''
        res = []
        if sentence is None or len(sentence) == 0:
            return res

        #re正则表达式区分汉语的序列 之后直接通过re来划分sentence
        re_han = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)", re.U)
        blocks = re_han.split(sentence)
        # print(blocks)
        for blk in blocks:
            if not blk:
                continue

            if re_han.match(blk):
                divide = []
                prob, pos_list = self.__viterbi(blk)
                begin_, next_ = 0, 0
                for i, char in enumerate(blk):
                    pos = pos_list[i]
                    if pos == 'B':
                        begin_ = i
                    elif pos == 'E':
                        divide.append(blk[begin_:i + 1])
                        next_ = i + 1
                    elif pos == 'S':
                        divide.append(char)
                        next_ = i + 1
                if next_ < len(blk):
                    divide.append(blk[next_:])
                #print(divide)
                for word in divide:
                    res.append(word)
            else:
                res.append(blk)
        return res

def main1():
    train_file = 'data\\pku_training.utf8'
    segment = HMMSegment()
    segment.train([line for line in open(train_file, 'r', encoding='utf-8')])

    fw = open("data\\seg_result\\hmm_out.txt", 'w', encoding="utf-8")
    with open("data\\seg_result\\pku_test.utf8", 'r', encoding="utf-8") as f:
        for line in f.readlines():
            # print(line)
            # print("{}".format(segment.cut(line)))
            lineres = segment.cut(line)
            for word in lineres:
                if word[-1] != '\n':
                    fw.write(word + " ")
                else:
                    fw.write(word)

def main2():
    start_time = time.time()
    main1()
    elapsed_time = time.time() - start_time
    print("耗时{}s".format(elapsed_time))

if __name__ == '__main__':
    main1()