# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:19:26 2017

@author: ASY

stack 第二阶段，将第一阶段的tfidf+lr的结果和doc2vec+nn的结果作为特征输入xgboost
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from queryDataPro import dataPropre

def xgb_acc_score(preds,dtrain):
    y_true = dtrain.get_label()
    y_pred = np.argmax(preds,axis=1)
    return [('acc', np.mean(y_true == y_pred))]

tfidf_lr = pd.read_csv(r'D:\DPIDataAnalysis\data\tfidf_stack.csv')
doc2vec_nn = pd.read_csv(r'D:\DPIDataAnalysis\data\doc2vec_res.csv')
'''
concat几个数据集
'''
#df = pd.concat([tfidf_lr,doc2vec_nn],axis = 1)
df = doc2vec_nn

alldata = dataPropre()
ys = np.array(alldata['agePd'])   # label
num_class = len(pd.value_counts(ys))

# 取用全部数据中的前12000条记录作为训练数据，后面的作为验证数据
trainNum = 12000
X = df.iloc[:trainNum]
y = ys[:trainNum] 
X_te = df.iloc[trainNum:]
y_te = ys[trainNum:]

'''
新建一个存放预测结果的dataframe
'''
df_sub = pd.DataFrame()
df_sub['ip'] = alldata.iloc[trainNum:].index

seed = 10
esr = 100  # 在100次迭代中准确率如果没有提升就停止训练
evals = 1
ss = 0.9
mc = 2
md = 8
gm = 2
n_trees = 100  # 使用30棵树

params = {
          "objective":"multi:softprob",  # 定义学习任务及相应学习目标，输出的是[ndata,nclass]的向量，每行数据表示样本所属于每个类别的概率
          "booster":"gbtree",  # gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算
          "num_class":num_class,  # 处理多分类问题
          "max_depth":md,  # 树的最大深度，缺省值为6
          "min_child_weight":mc,   # 孩子节点中最小的样本权重和。调大这个参数能够控制过拟合
          "subsample":ss,   # 用于训练模型的子样本占整个样本集合的比例。如果设置为0.5意味着XGBoost将随机从样本集合中抽取50%的子样本建立树模型，相当于1-dropout，可以防止过拟合
          "colsample_bytree":0.8,   # 在建立树的时候对特征随机采样的比例
          "gamma":gm,  # 对一个节点的划分只在其loss function得到结果大于0的情况下进行，而gamma给定了所需的最低loss function的值
          "eta":0.01,  # 为了防止过拟合，在更新过程中使用的收缩步长，在每次提升计算之后，算法会直接获得新特征的权重。eta通过缩减特征的权重使提升计算过程更加保守
          "lambda":0,  # l2正则的惩罚系数
          "alpha":0,  # l1正则的惩罚系数
          "silent":1,  # 在偏置上的L2正则
}
dtrain = xgb.DMatrix(X,y)  # 组合成xgb的输入数据格式
dvalid = xgb.DMatrix(X_te, y_te)
watchlist = [(dtrain,'train'),(dvalid,'eval')]
bst = xgb.train(params, dtrain, n_trees, evals = watchlist, feval = xgb_acc_score, maximize = True, early_stopping_rounds = esr, verbose_eval = evals) # verbose_eval打印结果
# maximize，feval最大化xgb_acc_score指标
df_sub['agePd'] = np.argmax(bst.predict(dvalid),axis = 1) + 1


