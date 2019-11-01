import requests
from bs4 import BeautifulSoup
import pandas as pd
#import csv
#import os
#import re
#import numpy as np
#import datetime
#from pandas_datareader import data
#from pandas import Series, DataFrame

#???如何获取时间 -> extractDate 
#提取日期
def extractDate(str):
    #print(str.find('2019-10-31'))
    if str.find('/2019-'):
        i = str.find('2019-')
        return str[i:i + 10]
    else:
        return None
    
#根据相关链接获取更多的新闻
#???简书当中的那个api接口有什么作用呢 如何更高效地进入更多网页获取更多新闻呢
def getMoreTitles(url, count, df):
    reqMore = requests.get(url = url)
    reqMore.encoding = 'utf-8'
    soup = BeautifulSoup(reqMore.text, 'html.parser')
    for news in soup.select('li'):
        if len(news.select('a')) > 0:
            time = extractDate(news.select('a')[0]['href'])
            title = news.select('a')[0].text
            link = news.select('a')[0]['href']
            #allNewsTitle.append(singleNewsTitle)
            df.loc[count] = [time, title, link]
            count += 1
    return count                   


#获取首页上出现的所有新闻标题
def getTitle(url, count, df):
    req = requests.get(url = url)
    req.encoding = 'utf-8'
    soup = BeautifulSoup(req.text, 'html.parser')
    for news in soup.find('li'):
        if len(news.select('a')) > 0:
            time = extractDate(news.select('a')[0]['href'])
            title = news.select('a')[0].text
            link = news.select('a')[0]['href']
            #print(singleNewsTitle[0], singleNewsTitle[1], singleNewsTitle[2])
            df.loc[count] = [time, title, link]
            count += 1
    return count
    #print(allNewsTitle)


count = 0
df = pd.DataFrame(columns = ['Time', 'Title', 'Link'])

if __name__ == '__main__':
    url = "https://finance.sina.com.cn/stock/"
    save_road = "C:\\Users\\AAA\\.spyder-py3\\crawler\\SinaNewsTitle.csv"
    
    getTitle(url, count, df)
	
    #深度研究
    for i in range(1,25):
        url = "http://finance.sina.com.cn/roll/index.d.html?lid=1008&page=%d" % i
        count = getMoreTitles(url, count, df)
        print(count)
    #公司观察
    for i in range(1,25):
        url = "http://finance.sina.com.cn/roll/index.d.html?cid=221431&page=%d" % i
        count = getMoreTitles(url, count, df)
        print(count)
    #大盘评述
    for i in range(1,25):
        url = "http://finance.sina.com.cn/roll/index.d.html?cid=56589&page=%d" % i
        count = getMoreTitles(url, count, df)
        print(count)
    #个股点评
    for i in range(1,25):
        url = "http://finance.sina.com.cn/roll/index.d.html?cid=56588&page=%d" % i
        count = getMoreTitles(url, count, df)
    #名家看市
    for i in range(1,15):
        url = "http://finance.sina.com.cn/roll/index.d.html?cid=57568&page=%d" % i
        count = getMoreTitles(url, count, df) 
        print(count)
    #板块个股
    for i in range(1,15):
        url = "http://finance.sina.com.cn/roll/index.d.html?cid=230808&page=%d" % i
        count = getMoreTitles(url, count, df)
        print(count)
    #实时解盘
    for i in range(1,15):
        url = "http://finance.sina.com.cn/roll/index.d.html?cid=57568&page=%d" % i
        count = getMoreTitles(url, count, df)
        print(count)
    #证券评论
    for i in range(1,15):
        url = "http://finance.sina.com.cn/roll/index.d.html?cid=57569&page=%d" % i
        count = getMoreTitles(url, count, df)
        print(count)
    #主力-基金
    for i in range(1,25):
        url = "http://finance.sina.com.cn/roll/index.d.html?cid=56615&page=%d" % i
        count = getMoreTitles(url, count, df) 
        print(count)
    #宏观分析
    for i in range(1,25):
        url = "http://finance.sina.com.cn/roll/index.d.html?cid=56598&page=%d" % i
        count = getMoreTitles(url, count, df) 
        print(count)
    #策略研究
    for i in range(1,25):
        url = "http://finance.sina.com.cn/roll/index.d.html?cid=56605&page=%d" % i
        count = getMoreTitles(url, count, df) 
        print(count)
    '''
    这一段当中的页面结构较之前有所不同 需要修改代码
    #公司-行业
    for i in range(1,1061):
        url = "http://stock.finance.sina.com.cn/stock/go.php/vReport_List/kind/company/index.phtml?p=%d" % i
        getMoreTitles(url, count, df)
        股市汇的内容未囊括其中
    '''
    #print(df)
    df.to_csv('SinaNewsTitle.csv', index = True)