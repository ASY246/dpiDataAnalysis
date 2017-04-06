# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:59:03 2017

@author: ASY

char-cnn，网络构建和结果输出部分
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

allWords = set()
for query_id,query in enumerate(query_all):
    for char in query:
        allWords.add(char)
    
allWords = list(allWords)

char2Index = {item:index for index, item in enumerate(allWords)}
index2Char = {index:item for index, item in enumerate(allWords)}
alphabet_size = len(allWords)
           
'''
每篇文本长度不相等，设窗口长度为20个字符，超过20个字符的部分截掉不要，不足20个字符的部分，填0，考虑使用协同过滤补齐？
'''
label_all = np.array(label_all).astype('int64')
window = 20  #window一改动后面的池化层等参数都要随之变化
allFeatureIndex = np.zeros((len(label_all),window))
for query_id, query in enumerate(query_all):
    for char_id, char in enumerate(query):
        if char_id < 20:
            allFeatureIndex[query_id][char_id] = char2Index[char]

allFeatureIndex = np.array(allFeatureIndex).astype('int64')

'''
这里省略shuffle步骤，直接取batch,定100个epoch
''' 
epoch_num = 30
batch_size = 256  #一个batch包含的样本点
NumOfBatches = int(np.ceil(len(label_all)/100))  # 一个epoch中需要取到的batch数目

def selectBatch(batch_size, batch_index, lengthDataSet = len(label_all)):
    '''
    传入batch的大小，batch的序号，数据集最大长度
    '''
    batchStartIndex = batch_index*batch_size
    batchEndIndex = min((batch_index + 1)*batch_size, lengthDataSet)
    
    batchFeature = allFeatureIndex[batchStartIndex:batchEndIndex]
    batchLabel = label_all[batchStartIndex:batchEndIndex]
# label转为one-hot编码
    batchLabel = pd.get_dummies(batchLabel)

    return batchFeature, batchLabel
    
"""--------------------------------------------------------------------------------------------------------------------------------"""
'''
使用tensorflow建图，网络参数
'''
conv_layers = [
    [256, 5, None]  # 每次卷积产生256个特征，卷积核宽度为5，池化层为3
#    [253, 5, 3],
#    [256, 3, None],
#    [256, 3, None],
#    [256, 3, None],
#    [256, 3, 3]
    ]
fully_layers = [1024, 1024]
num_of_classes = 3  #三个类别,年龄段类别

with tf.name_scope("Input_Layer"):
    '''
    两个输入数据，训练特征x，训练标签y
    '''
    input_x = tf.placeholder(tf.int64,shape = [None, window], name = 'input_x')     # 输入样本x的格式，[样本数量，x的长度（字符串加窗后的长度）]
    input_y = tf.placeholder(tf.int64, name = 'input_y')
    dropout_keep_prob = tf.placeholder(tf.float64, name = 'dropout_keep_prob') # 由于训练和测试阶段dropout不同，因此这里需要设成placeholder
    
with tf.name_scope("Embedding_Layer"), tf.device('/cpu:0'):
    '''
    将原始数据转换为one-hot编码格式送入训练集
    '''

    charOneHot_x = tf.one_hot(list(range(alphabet_size)), alphabet_size, 1.0, 0.0)

    x = tf.nn.embedding_lookup(charOneHot_x, input_x)
    x = tf.expand_dims(x, -1)  # samples_num, window, alphabeta_length, channel
    
    charOneHot_y = tf.one_hot(list(range(num_of_classes)), num_of_classes, 1.0, 0.0)
    y = tf.nn.embedding_lookup(charOneHot_y, input_y)
    
    with tf.Session() as sess:
        print(sess.run(y,feed_dict = {input_y:label_all}))
        
    with tf.Session() as sess:
        test = sess.run(x,feed_dict = {input_x:allFeatureIndex[0:20]})   ##-----------------问题
    
with tf.name_scope("CNN_Layers"):
    '''
    6个卷积层带着6个池化层
    '''
    for layer_index, layer in enumerate(conv_layers):
        filter_width = x.get_shape()[2].value  # 卷积和宽度为字母表长度（one-hot向量维度）
        filter_shape = [layer[1], filter_width, 1, layer[0]] # 卷积核长度，卷积核宽度，输入channel数目，输出channel数目
        stdv = 1/sqrt(layer[0]*layer[1])
        W = tf.Variable(tf.random_uniform(filter_shape, minval = -stdv, maxval = stdv), dtype  ='float32', name = 'W')
        b = tf.Variable(tf.random_uniform(shape = [layer[0]], minval = -stdv, maxval = stdv), dtype = 'float32', name = 'b')
        
        conv = tf.nn.conv2d(x, W, [1,1,1,1], "VALID", name = 'Conv') # valid算法填充
        
        x = tf.nn.bias_add(conv, b)
        
        if layer[2] is None:
            x = tf.transpose(x,[0,1,3,2]) # 将x列表中元素换位置
        else:
            pool = tf.nn.max_pool(x, ksize=[1,layer[-1],1,1], strides = [1, layer[-1], 1, 1], padding = 'VALID')
            x = tf.transpose(pool,[0,1,3,2])
            
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test = sess.run(conv,feed_dict = {input_x:allFeatureIndex[0:20]})
    
with tf.name_scope("Reshape_layer"):
    '''
    将二维数据打成一维，传入全连接层中处理
    '''
    vec_dim = x.get_shape()[1].value * x.get_shape()[2].value

    x = tf.reshape(x, [-1, vec_dim])

with tf.name_scope("FullConnected_layer"):
    '''
    全连接层
    '''
    fullConnectNodes = [vec_dim] + list(fully_layers)

    for layer_index, layer in enumerate(fully_layers):
        stdv = 1/sqrt(fullConnectNodes[layer_index])
        W = tf.Variable(tf.random_uniform([fullConnectNodes[layer_index],layer], minval = -stdv, maxval = stdv), dtype = 'float32', name = 'W')
        b = tf.Variable(tf.random_uniform(shape = [layer], minval = -stdv, maxval = stdv), dtype = 'float32', name = 'b')
        
        x = tf.nn.xw_plus_b(x, W, b)
        
        x = tf.nn.dropout(x, 0.5)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test = sess.run(x,feed_dict = {input_x:allFeatureIndex[0:20]})        
        
with tf.name_scope("OutputLayer"):
    '''
    输出层,全连接层到输出类别的过度
    '''
    stdv = 1/sqrt(fullConnectNodes[-1])
    
    W = tf.Variable(tf.random_uniform([fullConnectNodes[-1], num_of_classes], minval = -stdv, maxval = stdv), dtype = 'float32', name = 'W')
    b = tf.Variable(tf.random_uniform(shape=[num_of_classes], minval = -stdv, maxval = stdv), name = 'b')
    
    p_y_given_x = tf.nn.xw_plus_b(x, W, b)
    predictions = tf.argmax(p_y_given_x, 1)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test = sess.run(p_y_given_x,feed_dict = {input_x:allFeatureIndex[0:100]})        

with tf.name_scope('loss'):
    losses = tf.nn.softmax_cross_entropy_with_logits(labels = input_y, logits = p_y_given_x)
    loss = tf.reduce_mean(losses)
    
with tf.name_scope("Accuracy"):
    correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"), name = "accuracy")

'''
启动session，feed_dict输入数据，计算graph，训练好神经网络模型
使用k折交叉验证输出stack第一阶段结果
'''
sess = tf.InteractiveSession()

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

train_num = 12000
crossVali = 2

x_train, y_train = query_all[:train_num],label_all[train_num:]
x_vali, y_vali = query_all[:train_num],label_all[train_num:]

stack_tr = np.zeros((x_train.shape[0], 3))  # [n_samples, n_classes]
stack_vali = np.zeros((x_vali.shape[0], 3))

accuracy_stat = []
loss_stat = []

for k, (tr, va) in enumerate(KFold(len(y_train),n_folds=crossVali)):
    print("{}_stack".format(k))
    sess.run(tf.global_variables_initializer())
    x_tr_stack, y_tr_stack = x_train[tr], y_train[tr]  # 用这部分训练模型
    x_va_stack, y_va_stack = x_train[va], y_train[va]  # 用这部分构造stack
    for epoch_index in range(epoch_num):
        print("{}_epoch".format(epoch_index))
        # 这里每个epoch要shuffle一次data
        for batch_index in range(len(y_tr_stack)//batch_size):
            x_batch, y_batch = selectBatch(batch_size, batch_index, len(y_tr_stack))            
            x_batch = np.asarray(x_batch, dtype = 'int64')
            feed_dict = {input_x:x_batch, input_y:y_batch, dropout_keep_prob:0.5}
# 输入维度的影响
            _, get_loss, get_accuracy = sess.run([train_step, loss, accuracy],feed_dict)   

#            losses = sess.run(loss,feed_dict)
        print("accuracy_{}".format(get_accuracy))
        accuracy_stat.append(get_accuracy)
        print("loss_{}".format(get_loss))
        loss_stat.append(get_loss)
            

    x_va_stack,y_va_stack = selectBatch(len(y_va_stack), 0, len(y_va_stack))
    feed_dict = {input_x:x_va_stack, input_y:y_va_stack, dropout_keep_prob:1.0}
    predictions_stackTrain, get_loss, get_accuracy = sess.run([predictions, loss, accuracy],feed_dict)
    print("va_stack_accuracy_{}".format(get_accuracy))
    print("va_stack_loss_{}".format(get_loss))
    
    x_vali, y_vali = selectBatch(len(y_vali), 0, len(y_vali))
    feed_dict = {input_x:x_vali, input_y: y_vali, dropout_keep_prob:1.0}
    predictions_stackVali, get_loss, get_accuracy = sess.run([predictions, loss, accuracy],feed_dict)
    print("vali_accuracy_{}".format(get_accuracy))
    print("vali_loss_{}".format(get_loss))
#    stack_tr[tr] += predictions_stackTrain
#    stack_vali += predictions_stackVali
    
#stack_vali /= crossVali
#stack_all = np.vstack([stack_tr, stack_vali])

#df_stack = pd.DataFrame(index = range(len(query_all)))

#for i in range(stack_all.shape[1]):
#    df_stack['tfidf_lr_{}'.format(i)] = stack_all[:,i]

sess.close()    
#df_stack.to_csv(r'D:\DPIDataAnalysis\data\char_cnn_stack.csv', index = None, encoding = 'utf8')
    
    
    