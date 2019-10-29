
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

data_dir='C:\\Users\\18813\\Desktop\\project\\code'
fname=os.path.join(data_dir,'price2.csv')
df = pd.read_csv(fname)
print(df.head())
temp = df["Closed"].values
#要转化类型
float_data = df.values[:,1:].astype(np.float64)

mean=float_data[:2000].mean(axis=0)
float_data-=mean
std=float_data[:2000].std(axis=0)
float_data/=std

def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=512,step=1):
    if max_index is None:
        max_index=len(data)-delay-1
    i=min_index+lookback
    print(i)
    while 1:
        if shuffle:
            rows=np.random.randint(min_index+lookback,max_index,size=batch_size)
        else:
            if i+batch_size>=max_index:
                i=min_index+lookback
                

            rows=np.arange(i,min(i+batch_size,max_index))
            
            i+=len(rows)
            
        samples=np.zeros((len(rows),lookback//step,data.shape[-1]))
        targets=np.zeros((len(rows),))
        for j,row in enumerate(rows):
            indices=range(rows[j]-lookback,rows[j],step)
            samples[j]=data[indices]
            targets[j]=data[rows[j]+delay]
        yield samples,targets
        
lookback = 10 #10天
step = 1
delay = 1 #1天
batch_size = 512

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=3000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=3001,
                    max_index=3500,
                    step=step,
                    batch_size=batch_size)

test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=3500,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

val_steps = (3000 - 2001 - lookback) // batch_size 


test_steps = (len(float_data) - 3001 - lookback) // batch_size

model = Sequential()
model.add(layers.GRU(32,dropout=0.2,recurrent_dropout=0.2,input_shape=(lookback//step,float_data.shape[-1])))
model.add(layers.Dense(1))


model.compile(optimizer=RMSprop(),
              loss='mae',)



history = model.fit_generator(train_gen,
                              steps_per_epoch=10,  
                              epochs=5,
                              validation_data=val_gen,
                              validation_steps=val_steps
                              )

model.save("GRU_With_Dropout1.h5")


