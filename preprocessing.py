

import pynlpir
import csv
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
#with open('C:\\Users\\18813\\Desktop\\project\\SinaNewsTitle.csv','r')as csvfile:
    #reader=csv.reader(csvfile)
    #column1 = [row[1]for row in reader]
    #print(column1)
d = pd.read_csv('C:\\Users\\18813\\Desktop\\project\\News_Sentiment_Stock_Price_Prediction-master\\SinaNewsTitle.csv', usecols=['Title'])
print(d)
y = d['Title']
#print(y)
x = d.drop(['Title'], axis=1)
#print(x)
pynlpir.open()
pynlpir.open(encoding='utf-8')
m=[]
for i in range(1,6000):
    #print(pynlpir.segment(y[i]))
    m.append(pynlpir.get_key_words(y[i], weighted=True))
df=pd.DataFrame(m)
print(df)
df.to_csv(r'C:\Users\18813\Desktop\project\News_Sentiment_Stock_Price_Prediction-master\titlesegment.csv',index=True)
#f=open(r'C:\Users\18813\Desktop\project\News_Sentiment_Stock_Price_Prediction-master\titlesegment.csv','w',newline='',encoding='utf-8')

#csv_writer=csv.writer(f)

#csv_writer.writerow(['date','Closed'])


#for i in range(0,4934):
    #csv_writer.writerow([i,dfprice[0,i]])

#f.close()
