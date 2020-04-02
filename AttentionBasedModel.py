# -*- coding: utf-8 -*-

import csv
import cPickle
from copy import deepcopy
import jieba
jieba.load_userdict('wordDict.txt')  # 自定义词典
import numpy as np
import pandas as pd
from keras.models import *
from keras.optimizers import *
from keras.layers.core import *
from keras.layers import Input, Embedding, LSTM, Dense, merge, TimeDistributed
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras import backend as K
from keras.preprocessing import sequence
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Convolution1D, MaxPooling1D, GlobalAveragePooling1D,Input, Merge
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
np.random.seed(1337)


# 读取汽车品牌view
def readView():
    with open('View_new.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        column = [row[0] for row in reader]
    car_Brand = list(set(column))
    print '汽车品牌库包含 %s 个汽车品牌' % len(car_Brand)
    return car_Brand


# 读取测试集
def readTest():
    with open('Test.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        column0 = [row[0] for row in reader]
    sentenceID_test = [i.split('\t', 1)[0] for i in column0[1:]]  # 句子ID list
    content_test = [i.split('\t', 1)[1] for i in column0[1:]]     # 评论内容 list
    print '测试集有 %s 条句子' % len(content_test)
    test = [sentenceID_test, content_test]
    return test


# 读取训练集
def readtrain():
    with open('allTrain_includeView.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row for row in reader]
    content_train = [i[1] for i in column1[1:]]
    view_train = [i[2] for i in column1[1:]]
    opinion_train = [i[3] for i in column1[1:]]
    print '训练集有 %s 条句子' % len(content_train)
    train = [content_train, view_train, opinion_train]
    return train


# 将utf8的列表转换成unicode
def changeListCode(b):
    a = []
    for i in b:
        a.append(i.decode('utf8'))
    return a


# 对列表进行分词并用空格连接
def segmentWord1(cont):
    c = []
    for i in cont:
        a = list(jieba.cut(i))
        b = " ".join(a)
        c.append(b)
    return c


# 对列表进行分词用逗号连接
def segmentWord2(cont):
    c = []
    for i in cont:
        a = list(jieba.cut(i))
        c.append(a)
    return c


# 分句
def cut_sentence(words):      # 输入unicode，返回unicode
    start = 0
    i = 0  # 记录每个字符的位置
    if len(words) > 1:
        token = words[1]  # token用于分割连续标点的情况
    else: return words
    sents = []
    punt_list = ',.!?:;~，。！？：；～'.decode('utf8')  # 要解码为 unicode 才能进行匹配
    for word in words:
        if word in punt_list and token not in punt_list:
            sents.append(words[start:i+1])
            start = i + 1  # start标记到下一句的开头
            i += 1
        else:
            i += 1   # 若不是标点符号，则字符位置继续前移
            token = list(words[start:i+2]).pop()  # 取下一个字符,pop()移除并返回列表的最后一个元素
    if start < len(words):
        sents.append(words[start:])  # 这是为了处理文本末尾没有标点符号的情况
    return sents


word_dict = [line.strip() for line in open('wordDict.txt')]
word_dict = list(set(word_dict))
word_dict = changeListCode(word_dict)


# 获取分词
def getSeg(text):
    if not text:
        return ''
    if len(text) == 1:
        return text
    if text in word_dict:
        return text
    else:
        small = len(text) - 1
        text = text[0:small]
        return getSeg(text)


# 分词(正向最大匹配)
def word_Segmentation(test_str):   # 输入unicode,输出unicode
    test_str = test_str.strip()
    max_len = 20  # 正向最大匹配分词的最大长度
    result_str = []  # 保存要输出的分词结果
    result_len = 0
    while test_str:
        tmp_str = test_str[0:max_len]
        seg_str = getSeg(tmp_str)
        seg_len = len(seg_str)
        result_len = result_len + seg_len
        if seg_str.strip():
            result_str.append(seg_str)
        test_str = test_str[seg_len:]
    return result_str


def transLabel(labels):
    for i in range(len(labels)):
        if labels[i] == 'pos':
            labels[i] = 2
        elif labels[i] == 'neu':
            labels[i] = 1
        elif labels[i] == 'neg':
            labels[i] = 0
        else: print "label无效：",labels[i]
    return labels


def inverseTransLabel(labels):
    lab2 = []
    for i in range(len(labels)):
        if labels[i] == 2:
            lab2.append('pos')
        elif labels[i] == 1:
            lab2.append('neu')
        elif labels[i] == 0:
            lab2.append('neg')
        else:
            print "label无效：",labels[i]
    return lab2


# 主函数
def main():

    car_Brand = readView()
    car_Brand_unicode = changeListCode(car_Brand)

    test = readTest()
    sentenceID_test = test[0]
    content_test = test[1]
    content_test_unicode = changeListCode(content_test)

    train = readtrain()
    view_train = changeListCode(train[1])
    opinion_train = transLabel(train[2])


    #content_test = segmentWord2(content_test)
    '''

    result = {}
    for i in range(len(content_test_unicode)):
        sentence = []
        words = list(word_Segmentation(content_test_unicode[i]))
        for j in car_Brand_unicode:
            if j in words:
                view_sentence = []
                view_sentence.append(j)
                sen = deepcopy(content_test[i])
                view_sentence.append(sen)
                sentence.append(view_sentence)
        if i % 100 == 0:
            print i
        if len(sentence) == 0:
            continue
        result[sentenceID_test[i]] = sentence
    print '输出含有 %d 个ID' % len(result)
    output = [(key, value) for key, value in result.items()]
    output2 = [('SentenceId', 'View', 'Sentence')]
    for i in range(len(output)):
        a = [(output[i][0].encode('utf8'), j[0], j[1]) for j in output[i][1]]
        output2.extend(a)
    print '输出含有 %d 个视角情感对' % len(output2)


    cPickle.dump(output2, open("output2.pkl", "wb"))


    '''
    output2 = cPickle.load(open("output2.pkl", "rb"))

    sen_test = []
    for i in output2:
        sen_test.append(i[2])
    sen_test = sen_test[1:]

    view_test = []
    for i in output2:
        view_test.append(i[1])
    view_test = view_test[1:]


    sen_test = segmentWord2(sen_test)
    content_train = segmentWord2(train[0])

    w = []  # 将所有词语整合在一起
    for i in sen_test:
        w.extend(i)
    for i in content_train:
        w.extend(i)
    for i in view_test:
        w.append(i)
    for i in view_train:
        w.append(i)


    def get_aspect(X):
        ans = X[:, 0, :]
        return ans

    def get_context(X):
        ans = X[:, 1:, :]
        return ans

    def get_R(X):
        Y, alpha = X[0], X[1]
        ans = K.T.batched_dot(Y, alpha)
        return ans

    # 参数设置
    maxlen = 81
    epochs = 2
    batch = 32
    emb = 300
    lr = 0.001

    print('Preprocessing...')
    dict = pd.DataFrame(pd.Series(w).value_counts())  # 统计词的出现次数
    del w
    dict['id'] = list(range(1, len(dict) + 1))
    get_sent = lambda x: list(dict['id'][x])

    sent_train = pd.Series(content_train).apply(get_sent)
    for i in range(len(content_train)):  # 在第一个位置插入view的值，每个句子的第一个词为视角
        a = dict['id'][view_train[i]]
        sent_train[i].insert(0, a)

    sent_test = pd.Series(sen_test).apply(get_sent)
    for i in range(len(sen_test)):  # 在第一个位置插入view的值，每个句子的第一个词为视角
        a = dict['id'][view_test[i]]
        sent_test[i].insert(0, a)


    sent_train = list(sequence.pad_sequences(sent_train, maxlen=maxlen))
    sent_test = list(sequence.pad_sequences(sent_test, maxlen=maxlen))

    train_content = np.array(sent_train)
    train_opinion = np.array(opinion_train)
    train_opinion1 = np_utils.to_categorical(train_opinion, 3)
    test_content = np.array(sent_test)

    print('Build model...')

    main_input = Input(shape=(maxlen,), dtype='int32', name='main_input')
    x = Embedding(output_dim=emb, input_dim=len(dict) + 1, input_length=maxlen, name='x')(main_input)
    drop_out = Dropout(0.1, name='dropout')(x)
    w_aspect = Lambda(get_aspect, output_shape=(emb,), name="w_aspect")(drop_out)
    w_context = Lambda(get_context, output_shape=(maxlen - 1, emb), name="w_context")(drop_out)

    w_aspect = Dense(emb, W_regularizer=l2(0.01), name="w_aspect_1")(w_aspect)

    # hop 1
    w_aspects = RepeatVector(maxlen - 1, name="w_aspects1")(w_aspect)
    merged = merge([w_context, w_aspects], name='merged1', mode='concat')
    distributed = TimeDistributed(Dense(1, W_regularizer=l2(0.01), activation='tanh'), name="distributed1")(merged)
    flat_alpha = Flatten(name="flat_alpha1")(distributed)
    alpha = Dense(maxlen - 1, activation='softmax', name="alpha1")(flat_alpha)
    w_context_trans = Permute((2, 1), name="w_context_trans1")(w_context)
    r_ = merge([w_context_trans, alpha], output_shape=(emb, 1), name="r_1", mode=get_R)
    r = Reshape((emb,), name="r1")(r_)
    w_aspect_linear = Dense(emb, W_regularizer=l2(0.01), activation='linear')(w_aspect)
    merged = merge([r, w_aspect_linear], mode='sum')

    w_aspect = Dense(emb, W_regularizer=l2(0.01), name="w_aspect_2")(merged)

    # hop 2
    w_aspects = RepeatVector(maxlen - 1, name="w_aspects2")(w_aspect)
    merged = merge([w_context, w_aspects], name='merged2', mode='concat')
    distributed = TimeDistributed(Dense(1, W_regularizer=l2(0.01), activation='tanh'), name="distributed2")(merged)
    flat_alpha = Flatten(name="flat_alpha2")(distributed)
    alpha = Dense(maxlen - 1, activation='softmax', name="alpha2")(flat_alpha)
    w_context_trans = Permute((2, 1), name="w_context_trans2")(w_context)
    r_ = merge([w_context_trans, alpha], output_shape=(emb, 1), name="r_2", mode=get_R)
    r = Reshape((emb,), name="r2")(r_)
    w_aspect_linear = Dense(emb, W_regularizer=l2(0.01), activation='linear')(w_aspect)
    merged = merge([r, w_aspect_linear], mode='sum')

    w_aspect = Dense(emb, W_regularizer=l2(0.01), name="w_aspect_3")(merged)

    # hop 3
    w_aspects = RepeatVector(maxlen - 1, name="w_aspects3")(w_aspect)
    merged = merge([w_context, w_aspects], name='merged3', mode='concat')
    distributed = TimeDistributed(Dense(1, W_regularizer=l2(0.01), activation='tanh'), name="distributed3")(merged)
    flat_alpha = Flatten(name="flat_alpha3")(distributed)
    alpha = Dense(maxlen - 1, activation='softmax', name="alpha3")(flat_alpha)
    w_context_trans = Permute((2, 1), name="w_context_trans3")(w_context)
    r_ = merge([w_context_trans, alpha], output_shape=(emb, 1), name="r_3", mode=get_R)
    r = Reshape((emb,), name="r3")(r_)
    w_aspect_linear = Dense(emb, W_regularizer=l2(0.01), activation='linear')(w_aspect)
    merged = merge([r, w_aspect_linear], mode='sum')

    w_aspect = Dense(emb, W_regularizer=l2(0.01), name="w_aspect_4")(merged)

    # hop 4
    w_aspects = RepeatVector(maxlen - 1, name="w_aspects4")(w_aspect)
    merged = merge([w_context, w_aspects], name='merged4', mode='concat')
    distributed = TimeDistributed(Dense(1, W_regularizer=l2(0.01), activation='tanh'), name="distributed4")(merged)
    flat_alpha = Flatten(name="flat_alpha4")(distributed)
    alpha = Dense(maxlen - 1, activation='softmax', name="alpha4")(flat_alpha)
    w_context_trans = Permute((2, 1), name="w_context_trans4")(w_context)
    r_ = merge([w_context_trans, alpha], output_shape=(emb, 1), name="r_4", mode=get_R)
    r = Reshape((emb,), name="r4")(r_)
    w_aspect_linear = Dense(emb, W_regularizer=l2(0.01), activation='linear')(w_aspect)
    merged = merge([r, w_aspect_linear], mode='sum')

    w_aspect = Dense(emb, W_regularizer=l2(0.01), name="w_aspect_5")(merged)

    # hop 5
    w_aspects = RepeatVector(maxlen - 1, name="w_aspects5")(w_aspect)
    merged = merge([w_context, w_aspects], name='merged5', mode='concat')
    distributed = TimeDistributed(Dense(1, W_regularizer=l2(0.01), activation='tanh'), name="distributed5")(merged)
    flat_alpha = Flatten(name="flat_alpha5")(distributed)
    alpha = Dense(maxlen - 1, activation='softmax', name="alpha5")(flat_alpha)
    w_context_trans = Permute((2, 1), name="w_context_trans5")(w_context)
    r_ = merge([w_context_trans, alpha], output_shape=(emb, 1), name="r_5", mode=get_R)
    r = Reshape((emb,), name="r5")(r_)
    w_aspect_linear = Dense(emb, W_regularizer=l2(0.01), activation='linear')(w_aspect)
    merged = merge([r, w_aspect_linear], mode='sum')

    w_aspect = Dense(emb, W_regularizer=l2(0.01), name="w_aspect_6")(merged)

    # hop 6
    w_aspects = RepeatVector(maxlen - 1, name="w_aspects6")(w_aspect)
    merged = merge([w_context, w_aspects], name='merged6', mode='concat')
    distributed = TimeDistributed(Dense(1, W_regularizer=l2(0.01), activation='tanh'), name="distributed6")(merged)
    flat_alpha = Flatten(name="flat_alpha6")(distributed)
    alpha = Dense(maxlen - 1, activation='softmax', name="alpha6")(flat_alpha)
    w_context_trans = Permute((2, 1), name="w_context_trans6")(w_context)
    r_ = merge([w_context_trans, alpha], output_shape=(emb, 1), name="r_6", mode=get_R)
    r = Reshape((emb,), name="r6")(r_)
    w_aspect_linear = Dense(emb, W_regularizer=l2(0.01), activation='linear')(w_aspect)
    merged = merge([r, w_aspect_linear], mode='sum')

    h_ = Activation('tanh')(merged)
    out = Dense(3, activation='softmax')(h_)
    output = out
    model = Model(input=[main_input], output=output)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam(),
                  metrics=['accuracy'])

    print('Train...')
    model.fit(train_content, train_opinion1,
              batch_size=batch, nb_epoch=epochs
              )# validation_split=0.1

    scores = model.predict(test_content, batch_size=batch)
    predicted = []
    for i in range(scores.shape[0]):
        l = np.argmax(scores[i])
        predicted.append(l)

    predicted = inverseTransLabel(predicted)

    for i in range(len(predicted)):
        predicted[i] = predicted[i].encode('utf8')

    for i in range(len(view_test)):
        view_test[i] = view_test[i].encode('utf8')

    for i in range(1,len(output2)):
        output2[i] = (output2[i][0], view_test[i - 1], predicted[i - 1])

    csvfile = file('Output.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerows(output2)
    csvfile.close()


if __name__ == '__main__':
    main()
