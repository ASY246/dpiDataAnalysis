# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 08:26:16 2017

@author: ASY

使用朴素贝叶斯分类器完成第一层stack
"""

import tensorflow as tf
import numpy as np
from math import sqrt
import pandas as pd
from queryDataPro import dataPropre
from sklearn.cross_validation import KFold
'''
对所有汉字字符编码，使用与char-rnn中文文本生成相同的方式
'''

query_all = dataPropre()['queries'].values
label_all = dataPropre()['agePd'].values

'''
求P(Y),key为对应类别label
'''
value_counts = pd.value_counts(label_all)
p_y = {}
for i in value_counts.index:
    p_y[i] = value_counts[i]/len(label_all)
    
'''
求似然函数P(X|Y)，根据独立分布假设
'''
p_xy = {}
label_prob_all = np.zeros((len(pd.value_counts(label_all)), len(label_all)))
for i in value_counts.index:
    '''
    为计算所有词的概率乘积，对每个类别统计类别中所有的词汇出现的概率
    1.计算这个类别有哪些词，不含重复使用set
    2.分别计算这些词的概率，包含重复
    '''
    allWords = []
    for sample_index, label in enumerate(label_all):
        if label == i:  # 对于这个类别
            for word in query_all[sample_index].split(' '):
                allWords.append(word)
                
    allWordsLength = len(allWords) #计算这一类的所有词长度，分母
    allDistinctWords = list(set(allWords))

    # 将该类别的所有query合并，使用pd.value_counts获取该类别中每个单词出现的频率
    WordsProb = pd.value_counts(allWords)/allWordsLength  # 这一类别的所有词的频率，似然函数
    
    # 对所有样本，计算分别对应三个标签的似然函数和先验分布的乘积
#    for feature, label in zip(query_all,label_all):
#        print("33")
    label_prob = {}
    for query_index, query in enumerate(query_all):
        words = query.split(' ')
        mulRes = 1
        for word in words:
            mulRes *= WordsProb.get(word, default = 1e-6)  # 如果这一类别没有这个词怎么办
        print(mulRes)        
        label_prob[query_index] = mulRes

        label_prob_all[i][query_index] = mulRes 


'''
似然函数与先验概率相乘，比较结果
'''
predRes = {}
label_prob_all_trans = label_prob_all.T
        

for sample_index, sample in enumerate(label_prob_all_trans):
    label_jd = []
    for prob_index, prob in enumerate(sample):
        label_jd.append(prob * p_y[prob_index])
    predRes[sample_index] = np.argmax(label_jd)