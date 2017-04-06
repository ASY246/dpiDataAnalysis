# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 09:38:42 2017

@author: ASY

文本向量转换
通过DM/DBOW两种方式提取Doc2Vec向量
"""

from datetime import datetime
import numpy as np
import pandas as pd
from queryDataPro import dataPropre

#def inputAllData():
#    '''
#    导入预处理之后的数据
#    '''
#    df_all = pd.read_csv(r'D:\DPIDataAnalysis\data\alldata.txt')
#    return df_all

def splitWords(df_all):
    '''
    将每个查询分词，相同用户的所有记录放在一起
    '''
    doc_f = open(r'D:\DPIDataAnalysis\data\alldata_id.txt','w',encoding='utf8')
    
    for i, queryWord in enumerate(df_all['queries']):
        doc_f.write('_*{} {}'.format(i,' '.join(queryWord)))    
    doc_f.close()

'''
------------------------------------------------------doc2vec训练文本向量--------------------------------------
'''
def doc2vec():
    '''
    对所有用户query做输入格式处理，训练doc2vec向量，最后采用mikolov原论文中推荐的方式，将dm和dbow向量contact后输出
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
    '''
    dbow模式，向量维度为300，删掉5个噪声词，不使用层次softmax，忽略所有出现频次低于3的词，窗口长度为30，高频词被随机降采样，8个线程训练，初始学习速率为0.025，最小学习速率为0.025
    '''            
    dbow = Doc2Vec(dm = 0, size = 300, negative = 5, hs = 0, min_count = 3, window = 30, sample = 1e-5, workers = 8, alpha = 0.025, min_alpha = 0.025)
    doc_list = Doc_list(r'D:\DPIDataAnalysis\data\alldata_id.txt')
    dbow.build_vocab(doc_list)
    
    # --------------------train dbow(bag of words) doc2vec
    for i in range(2):
        print(datetime.now(), 'pass:', i + 1)
        dbow.train(doc_list)    
    dbow.save(r'D:\DPIDataAnalysis\model\dbow_d2v.model')
    print(datetime.now(), 'dbow_d2v save done')    
    '''
    dm模型，alpha为0.05，window=10
    '''
    dm = Doc2Vec(dm=1, size = 300, negative = 5, hs = 0, min_count = 3, window = 10, sample = 1e-5, workers = 8, alpha = 0.05, min_alpha = 0.025)
    doc_list = Doc_list(r'D:\DPIDataAnalysis\data\alldata_id.txt')
    dm.build_vocab(doc_list)    
    for i in range(10):
        print(datetime.now(),"pass",i)
        doc_list = Doc_list(r'D:\DPIDataAnalysis\data\alldata_id.txt')
        dm.train(doc_list)
        
    dm.save(r'D:\DPIDataAnalysis\model\dm_d2v.model')
    print(datetime.now(),'db_d2v save done')
    
    return dbow,dm

    
def doc2vecRes():
    df_all = dataPropre()
    splitWords(df_all)
    dbow,dm = doc2vec()
#    df_all['queryVec_dbow'] = dbow.docvecs
#    df_all['queryVec_dm'] = dm.docvecs
    return dbow,dm

    
    
    
    
    
    
    
    
    
    
    
    
    
    