from genericpath import exists
import pandas as pd
import numpy as np
import os
import random
import torch

#datapath='1'

def add_val_to_train(loaddatapath,savedatapath,randrate):
    train_data = torch.load(loaddatapath+'/'+'train.pt')
    val_data = torch.load(loaddatapath+'/'+'val.pt')
    save_testdata = torch.load(loaddatapath+'/'+'test.pt')
    print("训练集大小",len(train_data))
    print("验证集大小",len(val_data))
    #print("训练集",train_data)
    #print("测试集",val_data)
    vallength=np.arange(0,len(val_data),1)
    #print(vallength)
    random.seed(2022)
    random.shuffle(vallength)
    rand_vallen = vallength
    #print(rand_vallen)

    x=np.floor((randrate*len(val_data)-len(train_data))/(1+randrate))
    print("num", x)
    print("numt", x+len(train_data))
    print("numv", len(val_data)-x)

    rand_vallen=np.array(rand_vallen,dtype='int')
    #print(rand_vallen)

    shuffled_valdata = [val_data[i] for i in rand_vallen]
    for i in range(0,int(x)):
        train_data.append(shuffled_valdata[i])

    save_train = train_data


    if len(train_data)>10000:
        index_train = np.arange(0, len(train_data))
        # j进行undersample
        random.seed(2022)
        random.shuffle(index_train)
        train_data_under = [train_data[i] for i in index_train]
        runs=int(len(train_data_under)/10)
        train_under=train_data_under[0:runs]
        print(len(train_under))
        save_train =train_under

    if len(train_data)<2000:
        index_train=np.arange(0,len(train_data))
        #j进行oversample
        random.seed(2022)
        random.shuffle(index_train)
        train_data_underover = [train_data[i] for i in index_train]
        runs=int(2000/len(train_data))
        train_over=[]
        for t in range(0,runs):
            for j in range(0,len(train_data_underover)):
                train_over.append(train_data_underover[j])
        for t in range(0,int(2000%len(train_data_underover))):
            train_over.append(train_data_underover[t])
        print(len(train_over))
        save_train =train_over



    print("训练集大小",len(save_train))

    save_val=shuffled_valdata[int(x):]
    #print(save_val)
    print("验证集大小", len(save_val))
    print("save---")
    torch.save(save_train,savedatapath+'/'+'train.pt')
    torch.save(save_val,savedatapath+'/'+'val.pt')
    torch.save(save_testdata, savedatapath + '/' + 'test.pt')


import os
os.makedirs('../cikmdata_v2/CIKM22Competition',exist_ok=True)
for i in range(0,13):
    os.makedirs(f'../cikmdata_v2/CIKM22Competition/{i+1}',exist_ok=True)

#randrate是train val比例
randrate=8
for i in range(0,13):
    savedatapath=f'../cikmdata_v2/CIKM22Competition/{i+1}'
    datapath=f'./data/CIKM22Competition/{i+1}'
    print("客户端",datapath)
    add_val_to_train(datapath,savedatapath,randrate)



