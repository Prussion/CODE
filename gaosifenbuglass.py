from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.stats import multivariate_normal

latent_variable_dim= 90
class_num = 10
def gaosi_chubu(Yy1, sample, mnists):
    mnists = np.array(mnists)
    meanmnist = [0] * class_num
    for i in range(class_num):
        meanmnist[i] = []
    for i in range(sample):
        for j in range(class_num):
            if mnists[i] == j:
                meanmnist[j].append(Yy1[i, :])
                break


    meantrain = []
    for i in range(class_num):
        meantrain.append(np.mean(meanmnist[i], axis=0))
    lovtrain = []
    for i in range(class_num):
        lovtrain.append(np.cov(meanmnist[i], rowvar=0).reshape(-1))



    lovtrain = np.array(lovtrain)
    meantrain = np.array(meantrain)


    return lovtrain, meantrain


a1=np.eye(latent_variable_dim,dtype=int)
def gaosizhengtaifenbu(Yy1,Ymean_train1,Y_train_cov1):
    vt=[]
    vs=[]
    a=np.eye(latent_variable_dim)*0.01
    for i in range(class_num):
        vv = []
        Y_train_coV1 = Y_train_cov1[i,:].reshape(latent_variable_dim, latent_variable_dim) + 0.01*np.eye(latent_variable_dim, dtype=int)


        new = multivariate_normal.pdf(Yy1, Ymean_train1[i,:], Y_train_coV1)

        vt.append(new)


    vt = np.array(vt)
    Yy_hang = np.argmax(vt, axis=0)


    return Yy_hang
def diedai(index_train1,labels,Ymean_train,Yy1,Yy_hang,Y_train_cov1,len_train,sample,r1):#
        for i in index_train1:
            Yy1[i,:]=Yy1[i,:]-Ymean_train[labels[i],:]
            wufenlei = Yy1[i, :] -r1[int(labels[i]), :]

            wufenlei1 = np.dot(wufenlei.reshape(-1, 1), wufenlei.reshape(1, -1)).reshape(-1)

            wufenlei2 = Yy1[i, :] - r1[Yy_hang[i], :]
            f =labels[i]

            Y_train_coV1 = np.array(Y_train_cov1)
            wufenlei2 = np.dot(wufenlei2.reshape(-1, 1), wufenlei2.reshape(1, -1)).reshape(-1)
            Y_train_cov1[f, :] = (1 / (len_train[f] + 1)) * ((len_train[f] +1)* Y_train_cov1[f,:] + wufenlei1)
            Y_train_cov1[Yy_hang[i],:] = (1 / (len_train[Yy_hang[i]] - 1)) * (
                    ( len_train[Yy_hang[i]]-1) * Y_train_cov1[Yy_hang[i],:] - wufenlei2)
        return np.sum(Yy_hang ==labels) / sample

