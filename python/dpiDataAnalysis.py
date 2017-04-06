# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:14:53 2017

@author: ASY

数据从spark拿下来的预处理部分

"""

import pandas as pd
import jieba
from datetime import datetime
import numpy as np
'''
过滤加载数据
'''
list_raw = []
for i, line in enumerate(open(r"D:\DPIDataAnalysis\data\dpiAnalysis", encoding='utf8')):
    segs = line.split('\u0005')
    row = {}
    row['ip'] = segs[0]
    row['birthday'] = segs[1]
    row['education'] = segs[2]
    row['gender'] = segs[3]
    row['query'] = '\t'.join(segs[4].split('@@@'))

    list_raw.append(row)
    if i%200000 == 0:
        print(segs)

def str2age(string):
    '''
    将生日字符串转化为年龄，分三个阶段，18岁以下，50岁以上
    '''
    string = string[:8]
    year = int(string[:4])
    
    return 2017-year
    
df_raw = pd.DataFrame(list_raw)
df_raw = df_raw.set_index('ip')

df_raw['birthday'] = df_raw['birthday'].apply(str2age)

df_tr = df_raw.iloc[:12000]
df_vd = df_raw.iloc[12001:]

'''
训练模型Doc2Vec
'''
doc_f = open(r'D:\DPIDataAnalysis\data\alldata_id.txt','w',encoding='utf8')

for i, queries in enumerate(df_raw['query']):
    words = []
    for query in queries.split('\t'):
        words.extend(list(jieba.cut(query)))        
    tags = [i]
    if i % 10000 == 0:
        print(words)
    doc_f.write('_*{} {}'.format(i,' '.join(words)))
    
doc_f.close()

'''
训练doc2vec模型
'''
from collections import namedtuple
SentimentDocument = namedtuple('SentimentDocument','words tags')
from gensim.models import Doc2Vec
class Doc_list(object):
    def __init__(self, f):
        self.f = f
    def __iter__(self):
        for i, line in enumerate(open(self.f,encoding='utf8')):
            words = line.split()
            tags = [int(words[0][2:])]
            words = words[1:]
            yield SentimentDocument(words,tags)
            if i % 100000 == 0:
                print(tags)
'''
dbow模式，向量维度为300，删掉5个噪声词，不使用层次softmax，忽略所有出现频次低于3的词，窗口长度为30，高频词被随机降采样，8个线程训练，初始学习速率为0.025，最小学习速率为0.025
'''            
d2v = Doc2Vec(dm = 0, size = 300, negative = 5, hs = 0, min_count = 3, window = 30, sample = 1e-5, workers = 8, alpha = 0.025, min_alpha = 0.025)
doc_list = Doc_list(r'D:\DPIDataAnalysis\data\alldata_id.txt')
d2v.build_vocab(doc_list)

# --------------------train dbow(bag of words) doc2vec
for i in range(2):
    print(datetime.now(), 'pass:', i + 1)
    d2v.train(doc_list)
    X_d2v = np.array([d2v.docvecs[i] for i in range(100000)])

d2v.save(r'D:\DPIDataAnalysis\data\dbow_d2v.model')
print(datetime.now(), 'dbow_d2v save done')


'''
dm模型，alpha为0.05，window=10
'''
d2v = Doc2Vec(dm=1, size = 300, negative = 5, hs = 0, min_count = 3, window = 10, sample = 1e-5, workers = 8, alpha = 0.05, min_alpha = 0.025)
doc_list = Doc_list(r'D:\DPIDataAnalysis\data\alldata_id.txt')
d2v.build_vocab(doc_list)

for i in range(10):
    print(datetime.now(),"pass",i)
    doc_list = Doc_list('alldata-id.txt')
    d2v.train(doc_list)
    X_d2v = np.array([d2v.docvecs[i] for i in range(100000)])
    
d2v.save(r'D:\DPIDataAnalysis\data\dm_d2v.model')
print(datetime.now(),'db_d2v save done')
    

    
    
    
    