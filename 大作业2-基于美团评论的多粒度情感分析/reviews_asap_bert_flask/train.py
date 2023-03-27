#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from bert4keras.backend import keras, set_gelu, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding

from keras.layers import Lambda, Dense, Dropout
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm


# In[2]:


import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.24
sess=tf.compat.v1.Session(config=config)


# In[3]:


config_path = 'RoBERTa-tiny3L312-clue/bert_config.json'
checkpoint_path = 'RoBERTa-tiny3L312-clue/bert_model.ckpt'
dict_path = 'RoBERTa-tiny3L312-clue/vocab.txt'


# In[2]:


train_data = pd.read_csv('data/sentiment_analysis_trainingset.csv').iloc[:, 1:].dropna()
valid_data = pd.read_csv('data/sentiment_analysis_validationset.csv').iloc[:, 1:].dropna()


# In[5]:


train_data.head()


# In[5]:


train_data.iloc[:, 1:] = train_data.iloc[:, 1:].apply(lambda x: x + 2)
valid_data.iloc[:, 1:] = valid_data.iloc[:, 1:].apply(lambda x: x + 2)


# In[6]:


train_data.columns


# In[7]:


# for column in train_data.columns:
#     print(valid_data[column].value_counts())


# In[8]:


len(train_data)


# In[9]:


len(valid_data)


# In[10]:


tags = train_data.columns[1:]
tags


# In[11]:


num_classes_topic = len(tags)
num_classes_topic


# In[12]:


set_gelu('tanh')  # 切换gelu版本

maxlen = 256
batch_size = 128
epochs = 25
dropout_rate = 0.1
num_classes_sentiment = 4


# bert

# In[13]:


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


# In[14]:


def get_data(data):
    token_ids = []
    segment_ids = []
    label = []
    # 循环每个句子
    for text in tqdm(data['content'].astype(str)):
        # 分词并把token变成编号
        token_id, segment_id = tokenizer.encode(text, maxlen=maxlen)
        token_ids.append(token_id)
        segment_ids.append(segment_id)
    token_ids = sequence_padding(token_ids)
    segment_ids = sequence_padding(segment_ids)
    
    # 获取20个维度的标签
    for columns in tags:
        label.append(np.array(data[columns]).astype('uint8'))
    label = np.array(label) 
    return [token_ids, segment_ids], label


# In[15]:


train_input, train_label = get_data(train_data)
valid_input, valid_label = get_data(valid_data)


# In[16]:


train_label.shape


# In[17]:


len(train_input)


# In[18]:


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='bert',
    return_keras_model=False,
)

pooler = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

# 多任务学习
mutil_layers = []
for i in range(num_classes_topic):
    dropout = Dropout(dropout_rate)(pooler)
    preds = Dense(num_classes_sentiment, activation='softmax', kernel_initializer=bert.initializer, name='preds_{}'.format(i))(dropout)
    mutil_layers.append(preds)
    
model = keras.models.Model(bert.model.input, mutil_layers)
# model.summary()


# In[19]:


loss_dict = {}
loss_weights_dict = {}
for i in range(num_classes_topic):
    loss_dict['preds_{}'.format(i)] = 'sparse_categorical_crossentropy'
    loss_weights_dict['preds_{}'.format(i)] = 1.


# In[20]:


# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
# loss_weights表示每个任务的权重，可以看情况设置

# 设置分段线性学习率
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(loss=loss_dict,
              loss_weights=loss_weights_dict,
              optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
                    int((len(train_input[0]) // batch_size * epochs) * 0.2): 1,
                    int((len(train_input[0]) // batch_size * epochs) * 0.3): 0.1
              }),
              metrics=['accuracy'],
             )


# In[21]:


def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


# In[22]:


# 写好函数后，启用对抗训练只需要一行代码
adversarial_training(model, 'Embedding-Token', 0.5)


# In[26]:


def evaluate(x_trues, y_trues):
    golds_topic, preds_topic = [], []
    golds_sentiment, preds_sentiment = [], []

    y_preds = np.argmax(model.predict(x_trues, batch_size = 128, verbose=1), axis=-1)
    for y_pred, y_true in zip(y_preds.T, y_trues.T):
        golds_topic.append([1 if y > 0 else 0 for y in y_true])
        preds_topic.append([1 if y > 0 else 0 for y in y_pred])
        golds_sentiment.extend(y_true)
        preds_sentiment.extend(y_pred)
    print(classification_report(y_true=golds_topic, y_pred=preds_topic, target_names=tags, digits=4))
    print(classification_report(y_true=golds_sentiment, y_pred=preds_sentiment, target_names=['未提及', '负', '中', '正'], digits=4))
    f1_score_topic = f1_score(y_true=golds_topic, y_pred=preds_topic, average='micro')
    f1_score_sentiment = f1_score(y_true=golds_sentiment, y_pred=preds_sentiment, average='micro')
    f1_score_average = float(format((f1_score_topic + f1_score_sentiment) / 2, '.4f'))
    print("f1_score_average", f1_score_average, type(f1_score_average))
    return f1_score_average


# In[27]:


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_f1 = evaluate(valid_input, valid_label)
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            model.save_weights('model/best_model_tag_sentiment.weights')
            model_json = model.to_json()
            with open('model/best_model_tag_sentiment.json', 'w') as json_file:
                json_file.write(model_json)
        print(
            u'val_f1: %.5f, best_val_f1: %.5f\n' %
            (val_f1, self.best_val_f1)
        )


# In[28]:


evaluator = Evaluator()

model.fit(
    train_input, [train_label[i] for i in range(num_classes_topic)], 
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[evaluator]
)


# In[29]:

# 验证模型
model.load_weights('model/best_model_tag_sentiment.weights')
print(u'final valid f1: %05f\n' % (evaluate(valid_input, valid_label)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




