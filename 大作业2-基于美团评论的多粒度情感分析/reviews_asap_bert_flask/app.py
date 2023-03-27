from flask import Flask, render_template, request
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open

from keras.layers import Lambda, Dense
from keras.models import model_from_json

import numpy as np
import pandas as pd
from tqdm import tqdm

app = Flask(__name__)

maxlen = 256

##设置路径
dict_path = 'RoBERTa-tiny3L312-clue/vocab.txt'
model_json_path = 'model/best_model_tag_sentiment.json'
model_weights_path = 'model/best_model_tag_sentiment.weights'

target_names_cn = ["交通是否便利", "距离商圈远近", "是否容易寻找", "排队等候时间", "服务人员态度", "是否容易停车", "点菜/上菜速度", "价格水平", "性价比",
                   "折扣力度", "装修情况", "嘈杂情况", "就餐空间", "卫生情况", "分量", "口感", "外观", "推荐程度", "本次消费感受", "再次消费的意愿"]

sentiment_names_cn = ["情感倾向未提及", "负面情感", "中性情感", "正面情感"]

model = model_from_json(open(model_json_path).read())
model.load_weights(model_weights_path)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def get_data_test(data):
    token_ids = []
    segment_ids = []
    texts = []
    # 循环每个句子
    for text in data:
        # 分词并把token变成编号
        token_id, segment_id = tokenizer.encode(text, maxlen=maxlen)
        token_ids.append(token_id)
        segment_ids.append(segment_id)
        texts.append(text)
    token_ids = sequence_padding(token_ids)
    segment_ids = sequence_padding(segment_ids)
    return [token_ids, segment_ids], texts


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        # 1.接请求数据,存放到变量text中
        text = request.form.get('txt')
        txt = request.form.get('txt')
        # 2.predict
        predict_input, predict_texts = get_data_test([text])
        predict_preds = np.argmax(model.predict(predict_input, verbose=1), axis=-1)
        neg = []
        neu = []
        pos = []
        for pred in predict_preds.T:
            for i, j in enumerate(pred):
                if j == 1:
                    neg.append(target_names_cn[i])
                elif j == 2:
                    neu.append(target_names_cn[i])
                elif j == 3:
                    pos.append(target_names_cn[i])
        neg_result = "|".join(neg)
        neu_result = "|".join(neu)
        pos_result = "|".join(pos)
        print(neg_result)
        print(neu_result)
        print(pos_result)
        return render_template('index.html', txt=txt, neg=neg_result, neu=neu_result, pos=pos_result)


if __name__ == '__main__':
    app.run(threaded=False)
