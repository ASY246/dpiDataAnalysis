# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:14:53 2017

@author: ASY

数据预处理，包括分布式处理后的数据，取年龄，查询词分词，文本向量转换模型
"""

import pandas as pd
import jieba

'''
过滤加载数据
'''

def str2age(string):
    '''
    将生日字符串转化为年龄，分三个阶段，18岁以下，50岁以上
    '''
    string = string[:8]
    year = int(string[:4])
    age = 2017-year
    
    if age < 30:
        agePd = 0
    elif age <50:
        agePd = 1
    else:
        agePd = 2
    return agePd
    
def wordSplit(queries):
    '''
    对queries进行分词，每个用户的所有查询词组成一个文本
    '''
    words = []
    for query in queries.split('\t'):
        words.extend(list(jieba.cut(query)))
    
    return ' '.join(words)
    
def dataPropre():
    list_raw = []
    for i, line in enumerate(open(r"D:\DPIDataAnalysis\data\dpiAnalysis", encoding='utf8')):
        segs = line.split('\u0005')
        row = {}
        row['ip'] = segs[0]
        row['birthday'] = segs[1]
        row['education'] = segs[2]
        row['gender'] = segs[3]
        row['queries'] = '\t'.join(segs[4].split('@@@'))
    
        list_raw.append(row)
        if i%200000 == 0:
            print(segs)

    df_raw = pd.DataFrame(list_raw)
    df_raw = df_raw.set_index('ip')
    
    df_raw['birthday'] = df_raw['birthday'].apply(str2age)
    df_raw.rename(columns = {'birthday':'agePd'},inplace = True)

    df_all = df_raw
    df_all['queries'] = df_all['queries'].apply(lambda x: wordSplit(x))
    
    df_all.to_csv(r'D:\DPIDataAnalysis\data\alldata.csv')
    
    return df_all
    
if __name__ == '__main__':
    dataPropre()
    