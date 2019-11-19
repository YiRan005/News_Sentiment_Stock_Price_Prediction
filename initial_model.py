# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 2019

@author: YiRan Hu
"""

import os
import pandas as pd
import numpy as np
import keras
from keras import models
from keras import layers
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

#industry="mine"
#ths=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
#for th in ths:
def train():
    data_dir='C:\\Users\\18813\\Desktop\\project\\code'
    fname=os.path.join(data_dir,'price7'+'.csv')
    fname2=os.path.join(data_dir,'price7'+'.csv')
    model_path='models/'+'/optimal_model_'+'_text.h5'

    data=pd.read_csv(fname,encoding='gb2312')
    all_data=data.iloc[:,2:].fillna(0)   #选取位置为第二列以后的整列数据 将缺失值设为0
    all_data=np.array(all_data,dtype=float)
    data2=pd.read_csv(fname2,encoding='gb2312')
    all_data2=data2.iloc[:,2:].fillna(0)
    all_data2=np.array(all_data2,dtype=float)
    
    callbacks_list=[
           keras.callbacks.EarlyStopping(monitor='acc', patience=1,),
           keras.callbacks.ModelCheckpoint(filepath=model_path, 
                                         monitor='val_loss', save_best_only=True,)
    ]    
    lookback=8  #观察将会回溯12天
    step=4 #每14天一个数据点进行采样？
    batch_size=14
    bound=bound=len(all_data)*3//4
    
    def generator(data, lookback, step, min_index, max_index, batch_size):
        i=min_index+lookback
        while 1:
            if i+batch_size>=max_index:
                i=min_index+lookback
            rows=np.arange(i, min(i+batch_size, max_index), step)
            i += batch_size
            
            samples=np.zeros((len(rows), lookback, data.shape[1]-1))
            targets=np.zeros((len(rows), ))
            for j, row in enumerate(rows):
                indices=range(rows[j]-lookback, rows[j])
                samples[j]=data[indices,:data.shape[1]-1]
                targets[j]=data[rows[j]][data.shape[1]-1]
            yield samples, targets
    
    train_gen=generator(all_data, lookback=lookback, step=step, min_index=0, max_index=bound, batch_size=batch_size)
    val_gen=generator(all_data, lookback=lookback, step=step, min_index=bound+1, max_index=len(all_data), batch_size=batch_size)
    test_gen=generator(all_data2, lookback=lookback, step=step, min_index=0, max_index=len(all_data2), batch_size=batch_size)
    
    train_steps=(bound-lookback)//batch_size
    val_steps=(len(all_data)-bound+1-lookback)//batch_size
    test_steps=(len(all_data2)-lookback)//batch_size
    
    # dropout LSTM
    model=models.Sequential()
    model.add(layers.LSTM(64, input_shape=(None, all_data.shape[1]-1)))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(64,return_sequences=False))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer=RMSprop(),loss='binary_crossentropy',metrics=['acc'])
    history=model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=20, 
                                validation_data=val_gen, validation_steps=val_steps,
                                callbacks=callbacks_list)
    
    # plot the loss
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs=range(1, len(loss)+1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()
    
    model2=models.load_model(model_path)
    results=model2.evaluate_generator(test_gen, steps=test_steps)
    print(results)

train()
