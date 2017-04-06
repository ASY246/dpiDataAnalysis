# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 22:01:09 2017

@author: ASY

tfidf+2-gram抽取用户搜索文档向量+逻辑回归预测年龄（多分类）

doc2vec不用2-gram，是因为这种方法已经考虑了词序信息
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from queryDataPro import dataPropre
import pickle
import pandas as pd
import numpy as np



alldata = dataPropre()
ys = np.array(alldata['agePd'])


class Tokenizer():
    def __init__(self):
        self.n = 0
    def __call__(self,line):  # 看下python的__call__和__iter__
        tokens = []
        words = line.split(' ')
        for gram in [1,2]:
            for i in range(len(words) - gram + 1):
                tokens += ["_*_".join(words[i:i+gram])]
        print(len(tokens))
        return tokens
'''
token对数据的2-gram处理，min_df,max_df表示选取词上下限的频率
'''
tfv = TfidfVectorizer(tokenizer=Tokenizer(),min_df=3,max_df=0.95,sublinear_tf=True)
X_sp = tfv.fit_transform(alldata['queries'])
pickle.dump(X_sp, open(r'D:\DPIDataAnalysis\model\tfidf_feature','wb'))

trainNum = 12000
crossVali = 5
'''
原来验证集的数据，12000之后的，没有用来训练模型，但是也经过了predict的处理，特征发生了变化，用的是平均值
'''
X = X_sp[:trainNum]
y = ys[:trainNum]
X_te = X_sp[trainNum:]
y_te = ys[trainNum:]

num_class = len(pd.value_counts(ys))
stack = np.zeros((X.shape[0], num_class))  # [n_samples, n_classes]
stack_te = np.zeros((X_te.shape[0], num_class))

for i, (tr,va) in enumerate(KFold(len(y),n_folds=crossVali)):
    clf = LogisticRegression(C = 3)
    clf.fit(X[tr],y[tr])
    y_pred_va = clf.predict_proba(X[va])  # 预测的结果概率-----------------------这个结果直接就把多分类的分类结果输出出来了
    y_pred_te = clf.predict_proba(X_te)
    stack[va] += y_pred_va
    stack_te += y_pred_te
        
stack_te /= crossVali
stack_all = np.vstack([stack,stack_te])

df_stack = pd.DataFrame(index = range(len(alldata)))

for i in range(stack_all.shape[1]):
    df_stack['tfidf_lr_{}'.format(i)] = stack_all[:,i]
             
df_stack.to_csv(r'D:\DPIDataAnalysis\data\tfidf_stack.csv', index = None, encoding = 'utf8')
        