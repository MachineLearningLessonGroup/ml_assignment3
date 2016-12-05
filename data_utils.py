#-*-coding:utf-8-*-
import numpy as np
import os
import urllib
from typing import Dict

#下载数据:
def download(url):
    """如果文件尚未下载,则从url地址下载并返回文件名"""
    filename=url.split('/')[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url,filename)
    if os.path.exists(filename):
        print("文件{0}已准备完毕!".format(filename))
    else:
        raise Exception('文件'+filename+'尚未下载,请检查网络!')
    return filename

def dispose_train(train_data, dictionary):
    '''将数据数值化处理,切分为特征X与类别y'''
    data=[]
    for line in train_data:
        for i in range(6):
            data.append([dictionary[i][line[i]],dictionary[6][line[6]]])
    data=np.array(data)
    #print(data[:20])
    return data[:,0],data[:,1]

def dispose_test(test_data,dictionary):
    test=[]
    for line in test_data:
        t=[]
        for i in range(7):
            t.append(dictionary[i][line[i]])
        test.append(t)
    test=np.array(test)
    return test[:,:6],test[:,6]


