# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:54:52 2017

@author: ASY

DPI data Analysis
"""
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
import numpy as np

def getHttpTitle(url = 'http://www.3761.com/yule/html/317029.html'):
    '''
    输入一个url，返回根据url检索网页的html文本中的中文 title
    ''' 
    html = urlopen(url)
    soup = BeautifulSoup(html,"lxml")
    title = soup.findAll("title")
    if len(title) > 0:
        res = [title[i].get_text() for i in range(len(title))]
    else:
        res = title
        
    return res

def httpGet(url, headerChange = True):
    '''
    调用http协议get方法获取html，通过requests修改请求头,headerChange = True使用伪装请求头（修改userAgent为iphone）
    headerChange = False使用普通方式
    
    百度爬虫：
    Mozilla/5.0 (compatible; Baiduspider/2.0; +http://www.baidu.com/search/spider.html)

    百度移动版爬虫：
    Mozilla/5.0 (Linux;u;Android 4.2.2;zh-cn;) AppleWebKit/534.46 (KHTML,like Gecko) Version/5.1 Mobile Safari/10600.6.3 (compatible; Baiduspider/2.0; +http://www.baidu.com/search/spider.html)

    搜狗爬虫：
    Sogou web spider/4.0(+http://www.sogou.com/docs/help/webmasters.htm#07)

    Chrome浏览器user agent：
    Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.94 Safari/537.36

    iphone6 plus 微信IOS版本：
    Mozilla/5.0 (iPhone; CPU iPhone OS 8_4_1 like Mac OS X) AppleWebKit/600.1.4 (KHTML, like Gecko) Mobile/12H321 MicroMessenger/6.3.9 NetType/WIFI Language/zh_CN

    魅蓝note 2 微信Android版本：
    Mozilla/5.0 (Linux; Android 5.1; m2 note Build/LMY47D) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/37.0.0.0 Mobile MQQBrowser/6.2 TBS/036215 Safari/537.36 MicroMessenger/6.3.18.800 NetType/WIFI Language/zh_CN
    '''    

    if headerChange == True:
        session = requests.Session()
        
        headers = {"User-Agent":"Mozilla/5.0 (compatible; Baiduspider/2.0; +http://www.baidu.com/search/spider.html)"} #伪装成百度爬虫
        print("session")
        try:
            req = session.get(url, headers = headers, timeout=5)
        except:
            return 'url get failed'
        print('get_text')
        html = req.content   #text返回的是Unicode型的数据，而使用content返回的是bytes型的数据。在使用r.content的时候，自带encoding转换
        print("soup")
        soup = BeautifulSoup(html,"lxml")
        print("title")
        title = soup.findAll("title")   
        
        if len(title) > 0:
            res = [title[i].get_text() for i in range(len(title))][0]  # 当前只输出第一个title
        else:
            res = 'no Title'
        return res
        
    else:
        try:
            req = urlopen(url, timeout=10).read()
        except Exception as e:
            print(e)
        return req
def dataFilter():
    '''
    输入getHttpTitle返回的数据，返回title
    '''
    pass

def dataAnalysis():
    testData = pd.read_csv(r"D:\DPIDataAnalysis\data\htmltext.txt",error_bad_lines=False)
    
if __name__ =='__main__':
     
    col_names = ['account','appName','webName','searchKeyWord','terminalType','url','UA']
    dpiData = pd.read_table(r'D:\DPIDataAnalysis\data\jiangsu_dpi_2017020820_sample001',names = col_names)


#    dpiData['httpTitle'] = dpiData['url'].apply(lambda x: getHttpTitle(x))  #pandas函数变换添加一列的方法
    
    urlArray = dpiData['url'].values
    file = open(r"D:\DPIDataAnalysis\data\htmltext.txt","w")
    res = []

    for i in range(len(dpiData)):
        print(i)
        value = httpGet(urlArray[i])
        res.append(value)
        try:
            file.writelines(value)
        except:
            file.writelines('gbk decode failed')
        
        file.writelines('\n')
    file.close()
    resSeries = pd.Series(res)
    dpiData['Web_Title'] = resSeries
    dpiData.to_csv(r"D:\DPIDataAnalysis\data\resText.txt")
    test3 = dpiData[pd.notnull(dpiData.iloc[:,3]) & (dpiData.iloc[:,7] != 'no Title')]
    
