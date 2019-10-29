# -*- coding: utf-8 -*-


def get_stock_price(ticker,atdate,fromdate):
    
    #抓取美股股票价格
    from pandas_datareader import data
    price=data.DataReader(ticker,'yahoo',fromdate,atdate)

    #按日期排序，近期的价格排在前面
    sortedprice=price.sort_index(axis=0,ascending=False)


    #提取收盘价
    closeprice=sortedprice.Close

    #将Series类型的closeprice转换为ndarray类型的ndprice
    import numpy as np
    ndprice=np.asmatrix(closeprice,dtype=None)
    
    #返回收盘价序列，按日期降序排列
    return ndprice

if __name__=='__main__':
    #指定日期范围：20150101-20190324
    from datetime import datetime
    atdate=datetime(2019,3,24)
    fromdate=datetime(2015,1,1)
    
    ndprice=get_stock_price('TAL',atdate,fromdate)
    ndprice.shape


def make_price_sample(ndprice,n_nextdays=1,n_samples=240,n_features=20):
    
    
    #生成第一个标签样本：标签矩阵y(形状：n_samples x 1)
    import numpy as np
    y=np.asmatrix(ndprice[0,0])
    #生成第一个特征样本：特征矩阵X(形状：n_samples x n_features)    
    X=ndprice[0,n_nextdays:n_features+n_nextdays]
    
    #生成其余的标签样本和特征样本 
    for i in range(1,n_samples):
        y_row=np.asmatrix(ndprice[0,i])
        y=np.append(y,y_row,axis=0)
    
        X_row=ndprice[0,(n_nextdays+i):(n_features+n_nextdays+i)]
        X=np.append(X,X_row,axis=0)
    
    return X,y

if __name__=='__main__':
    #指定日期范围：20150101-20190324
    from datetime import datetime
    atdate=datetime(2019,3,24)
    fromdate=datetime(2015,1,1)
    
    ndprice=get_stock_price('TAL',atdate,fromdate)
    
    X,y=make_price_sample(ndprice,1,240,20)
    X,y=make_price_sample(ndprice,5,120,10)
#def save_to_csv()

import numpy as np
import pandas as pd
import datetime
from pandas_datareader import data
from pandas import Series, DataFrame
import csv
dfprice=get_stock_price('MSFT','10/28/2019','3/20/2000')
#print(dfprice)
print(dfprice.shape)
df=pd.DataFrame(dfprice)
#print(df.columns)
#print(df.index)
#print(df.head(5))
#df_csvsave = web.DataReader("600018.SS","yahoo",datetime.datetime(2019,1,1),datetime.date.today())
#print (df_csvsave)
df.to_csv(r'C:\Users\18813\Desktop\project\code\price.csv',index=True)

f=open(r'C:\Users\18813\Desktop\project\code\price2.csv','w',newline='',encoding='utf-8')

csv_writer=csv.writer(f)

csv_writer.writerow(['date','Closed'])


for i in range(0,4934):
    csv_writer.writerow([i,dfprice[0,i]])

f.close()
    








    
