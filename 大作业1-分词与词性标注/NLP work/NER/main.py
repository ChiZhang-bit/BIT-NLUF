from data import build_corpus
from LSTM import BiLSTM
import pickle

def extend_maps(word2id, tag2id):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    return word2id, tag2id

def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)

def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model

def bilstm_train(train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists, test_word_lists, test_tag_lists, word2id, tag2id):
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BiLSTM(vocab_size, 100, 300, out_size)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)
    save_model(bilstm_model, "bilstm.pkl")

    pred_tag_lists, test_tag_lists = bilstm_model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)

    return pred_tag_lists

def main():
    # 第一步 读取数据：
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train.txt")
    dev_word_lists, dev_tag_lists = build_corpus("dev.txt")
    test_word_lists, test_tag_lists = build_corpus("test.txt")

    # 第二步 训练Bi-LSTM:
    #LSTM模型训练需要再word2id tag2id中国加入标志<pad> <unk>
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id)


    lstm_pred = bilstm_train(
        train_word_lists, train_tag_lists,
        dev_word_lists, dev_tag_lists,
        test_word_lists, test_tag_lists,
        bilstm_word2id, bilstm_tag2id,
    )
    print(lstm_pred)

    #第三步 载入模型进行识别：
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id)
    bilstm_model = load_model("bilistm.pkl")
    target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                   bilstm_word2id, bilstm_tag2id)
    # print(target_tag_list)

    #第四步 将数据写入测试结果文件中
    tag_label = {"O": 0,
                 "B-PER": 1, "I-PER": 2,
                 "B-LOC": 3, "I-LOC": 4,
                 "B-ORG": 5, "I-ORG": 6
                 }
    tag_label_list = [i for i,j in tag_label.items()]
    fw = open("ans.txt", 'w', encoding = "utf-8")
    for word, taglist in zip(test_word_lists, target_tag_list):
        if taglist in tag_label:
            fw.write(word + "\t" + tag_label_list[taglist] + "\n")

    fw.close()

if __name__ == "__main__":
    main()
