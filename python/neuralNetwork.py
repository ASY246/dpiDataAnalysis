# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 22:08:15 2017

@author: ASY

dbow,dm concat + nn softmax 生成stack1的结果
"""

import numpy as np
from gensim.models import Doc2Vec
import pandas as pd
from queryDataPro import dataPropre
from sklearn.cross_validation import KFold
from datetime import datetime

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop


def myAcc(y_true, y_pred):
    '''
    计算预测的accuracy
    '''
    y_pred = np.argmax(y_pred, axis = 1)
    return np.mean(y_true == y_pred)

'''
导入特征数据和标签，dbow+dm拼接为特征，年龄段为标签
'''
dbow = Doc2Vec.load(r'D:\DPIDataAnalysis\model\dbow_d2v.model')
dm = Doc2Vec.load(r'D:\DPIDataAnalysis\model\dm_d2v.model')
alldata = dataPropre()

X_sp = np.array([np.append(dbow.docvecs[i],dm.docvecs[i]) for i in range(len(dbow.docvecs))])
ys = {}
ys['agePd'] = np.array(alldata['agePd'])

'''
前12000个样本为训练样本，5折交叉验证
'''
trainNum = 12000
crossVali = 5

X = X_sp[:trainNum]  # 训练集 + 验证集
X_te = X_sp[trainNum:]  # 测试集
y = ys['agePd'][:trainNum]
y_te = ys['agePd'][trainNum:]

'''
stack
'''
num_class = len(pd.value_counts(ys['agePd']))
stack = np.zeros((X.shape[0], num_class))
stack_te = np.zeros((X_te.shape[0], num_class))

for k, (tr, va) in enumerate(KFold(len(y),n_folds=crossVali)):
    print('{} corssVali:{}/{}'.format(datetime.now(),k+1,crossVali)) #显示第几折交叉验证
    
    X_train = X[tr]
    y_train = y[tr]
    X_va = X[va]
    y_va = y[va]
    X_test = X_te
    y_test = y_te
    
    X_train = X_train.astype('float32')
    X_va = X_va.astype('float32')
    X_test = X_test.astype('float32')
    Y_train = np_utils.to_categorical(y_train, num_class)
    Y_va = np_utils.to_categorical(y_va, num_class)
    Y_test = np_utils.to_categorical(y_test, num_class)
    
    '''
    建一个softmax神经网络用于分类
    '''
    model = Sequential()
    model.add(Dense(250, input_shape=(X_train.shape[1],)))  # 写入向量维度，后面为空表示多少样本数量都可以作为输入
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    model.add(Dense(num_class))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])
    '''
    参数选择，verbose表示每个epoch输出一行记录，shuffle=True表示在训练中打乱输入样本的顺序,注意validation_data填test数据，
    因为test才是真正的validation，上面的validation只是为了做stack生成二级特征用的
    '''
    history = model.fit(X_train, Y_train, shuffle=True, batch_size=128, nb_epoch=35, verbose = 2, validation_data=(X_test, Y_test))
    '''
    predict_proba按batch产生输入数据的类别预测结果
    '''
    y_pred_va = model.predict_proba(X_va)
    y_pred_te = model.predict_proba(X_te)
    
    print('va acc:', myAcc(y[va],y_pred_va))
    print('te acc:', myAcc(y_te,y_pred_te))
    
    stack[va] += y_pred_va
    stack_te += y_pred_te
    
stack_te /= crossVali #因为test每次交叉验证都加了一遍，所以这里面除以交叉验证次数取平均
stack_all = np.vstack([stack,stack_te])

df_stack = pd.DataFrame(index = range(len(stack_all)))

for index in range(stack_all.shape[1]):
    df_stack['NN_{}'.format(index)] = stack_all[:,index]

df_stack.to_csv(r'D:\DPIDataAnalysis\data\doc2vec_res.csv',encoding = 'utf8', index = None)
print(datetime.now(),'save doc2vec_res done')
    
    
    
    
    
    
    
    
    
    