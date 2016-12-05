#-*-coding:utf-8-*-
'''
Generate by Python3.5

'''
import numpy as np
import random
#import time
from sklearn.cross_validation import train_test_split
#from sklearn.datasets.base import Bunch
import data_utils
from sklearn.naive_bayes import GaussianNB

'''

原数据格式:
| 类别:
unacc, acc, good, vgood

| 特征:
buying:   vhigh, high, med, low.
maint:    vhigh, high, med, low.
doors:    2, 3, 4, 5more.
persons:  2, 4, more.
lug_boot: small, med, big.
safety:   low, med, high.

'''
#数据链接:
url="http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"


#进行数值化处理的dict:
buying_dict={'vhigh':0,'high':1,'med':2,'low':3}
maint_dict={'vhigh':4,'high':5,'med':6,'low':7}
doors_dict={'2':8,'3':9,'4':10,'5more':11}
persons_dict={'2':12,'4':13,'more':14}
lug_dict={'small':15,'med':16,'big':17}
safety_dict={'low':18,'med':19,'high':20}
label_dict={'unacc':0,'acc':1,'good':2,'vgood':3}
dictionary=[buying_dict,maint_dict,doors_dict,persons_dict,lug_dict,safety_dict,label_dict]
features=list(range(21))
num_features=len(features)

raw_data=data_utils.download(url)
data_set=np.loadtxt(raw_data,delimiter=",",dtype=bytes).astype(str)
train_data, test_data=train_test_split(data_set, test_size=0.3, random_state=None)
X,y=data_utils.dispose_train(train_data,dictionary)
test_x,test_y=data_utils.dispose_test(test_data,dictionary)

'''
#将数据集切分为训练集和测试集:
train_data, test_data,train_target,test_target=\
    train_test_split(X, y, test_size=0.3, random_state=0)
print(len(train_data),len(train_target))
input()
gnb = GaussianNB()
gnb.fit(train_data,train_target)
print(gnb.score(test_data,test_target))
'''
label_count=[0,0,0,0]
for i in y:
    label_count[i]+=1
a=np.argsort(label_count)

num_data=len(X) #数据数量;
print('numdata ',num_data)
num_label=4
p=[]
for _ in range(num_label):
    p.append(np.random.uniform(size=num_features))  #每个类别下各个取值的概率; 
p=np.array(p)
print('p ',p)
alpha=np.array(np.random.uniform(size=num_label))
print('alpha ',alpha)
gama=np.array(np.zeros((num_data,num_label)))

#循环直至收敛：
#while(1):
for round in range(100):
    print(round,'\n')
    for j in range(num_data):
        #print('j: ',j,'\n')
        for k in range(num_label):
            pkj=p[k][X[j]]
            gama[j,k]=alpha[k]*pkj
        sigma=gama[j].sum()
        #print(sigma)
        gama[j]=gama[j]/sigma  #更新gama[j][k];
        #print('gama_j',gama[j])
        #input()
        
    for k in range(num_label):
        #print('k: ',k)
        t=gama[:,k]
        alpha[k]=t.sum()/num_data  #更新alpha;
        #print('alpha ',alpha)
        for m in range(num_features):
            #print('m: ',m)
            sum=0.0
            for j in range(num_data):
                if X[j]==features[m]:
                    sum+=gama[j,k]
            p[k][m]=sum
        sigma=p[k].sum()
        p[k]=p[k]/sigma   #更新p


b=np.argsort(alpha)
map_f=[0,0,0,0]
for i,j in zip(b,a):
    map_f[i]=j

#print('alpha:\n',alpha,'p\n\n',p,'\ngama\n',gama)
#分类
def naive_bayes(x,num_label,alpha,features):
    labels=[]
   
    for k in range(num_label):
        t=alpha[k]
        for i in x:
            if i==1:
                t=t*p[k,features[i]]
        labels.append(t)
    print('labels:\n',labels)
    #input('labels')
    return labels.index(max(labels))

the_labels=[]
for x in test_x:
    label=naive_bayes(x,num_label,alpha,features)
    the_labels.append(label)
correct=0
for i in range(len(the_labels)):
    if map_f[the_labels[i]]==test_y[i]:correct+=1
print(the_labels)
print('数目：',correct,'正确率：',correct/len(the_labels))
print('alpha:\n',alpha,'p\n\n',p,'\ngama\n',gama)








