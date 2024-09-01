import pandas as pd  # 数据科学计算工具
import gaosifenbuglass


from scipy.spatial.distance import pdist
from sklearn import datasets

import tensorflow as tf
import tensorflow.compat.v1 as tf
import math
import numpy as np
import time

from sklearn.model_selection import StratifiedKFold

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

start = time.time()
tf.disable_v2_behavior()
sess = tf.Session()

data = []

mnist = pd.read_csv('mnist_train.csv',header=None)

X = np.array(mnist.iloc[:, 1:])

Y = np.array(mnist.iloc[:, 0])

s = 1
feature = 784
class_num = 10
a = [1]
zhuchengfen = 120
latent_variable_dim = 90

v = np.random.rand(latent_variable_dim, latent_variable_dim)
sum1 = 0
sum2 = 0
sum3 = 0
sum4 = 0


def classical_gs(A):
    dim = A.shape

    Q = np.zeros(dim)  # initialize Q
    R = np.zeros((dim[1], dim[1]))  # initialize R
    for j in range(dim[1]):
        y = np.copy(A[:, j])
        for i in range(j):
            R[i, j] = np.matmul(np.transpose(Q[:, i]), A[:, j])
            y -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(y)
        Q[:, j] = y / R[j, j]
    return Q


def basic_pedcc(class_num, latent):
    a = [1]
    a1 = [-1]
    zero = np.zeros(latent - class_num + 1)
    zero = zero.tolist()

    a = a[:0] + zero + a[0:]
    a1 = a1[:0] + zero + a1[0:]

    u = np.stack((a1, a))
    u = u.tolist()


    for i in range(class_num - 2):
        # print(len(u))


        c = np.insert(u[len(u) - 1], 0, 0)

        for j in range(len(u)):

            p = np.append(u[j], 0).tolist()

            s = len(u) + 1
            u[j] = math.sqrt(s * (s - 2)) / (s - 1) * np.array(p) - 1 / (s - 1) * np.array(c)

        u.append(c)

    u = np.array(u)
    return u




r1 = np.dot(basic_pedcc(class_num, latent_variable_dim), classical_gs(v))


a2 = 0.1 * np.eye(zhuchengfen * 2 + 1, dtype=int)


def tanhni(s):
    uu = (2 / (1 - s)) - 1
    s = np.log(uu) / 2


    return s


def tanhdao(x):
    return 1 - ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))) ** 2


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def threshold(Yy1, YY, x_train, y_train, r33):
    class100 = [0] * class_num  # 对应c0,c1,c2

    sample_num = [0] * class_num
    num_smallwhole = []
    num_bigwhole = []
    for i in range(class_num):
        sample_num[i] = []  # 对应sample_num0, ...
    count2 = 0
    count = 0

    Z_100 = [0] * class_num
    for i in range(class_num):
        Z_100[i] = []

    print(r33.shape)
    for j in range(class_num):
        for i in range(0, len(Yy1)):
            z0 = np.dot(r33[j], Yy1[i])
            Z_100[j].append(z0)

    norm_100 = [0] * class_num
    for i in range(class_num):
        norm_100[i] = np.linalg.norm(r33[i])  # 对应abcd..
    norm_100s = [0] * class_num
    for i in range(class_num):
        norm_100s[i] = norm_100[i] * norm_100[i]

    for j in range(class_num):
        for i in range(len(y_train)):
            if y_train[i] == j:
                sample_num[j].append(i)
                class100[j] += 1
    throld = []
    for i in range(len(Yy1)):
        throld1 = []
        for j in range(class_num):
            throld1.append(np.dot(Yy1[i], r33[j]) / norm_100s[j])
        throld.append(throld1)
    throld = np.array(throld)

    for i in range(len(Yy1)):
        # print(throld[i])
        if all(throld[i] < 1):
            # print('1111',throld[i])
            num_smallwhole.append(i)
        elif any(throld[i] > 1):
            # print('2222',throld[i])
            num_bigwhole.append(i)

    print('larger', np.array(num_smallwhole).shape)
    print('smaller', np.array(num_bigwhole).shape)

    XU = []
    xU = []
    NUM = []
    NUM1 = []
    Y = []
    Y1 = []
    XU1 = []
    xU1 = []
    for i in num_smallwhole:
        XU.append(list(Yy1[i, :]))
        xU.append(list(x_train[i, :]))
        NUM.append(y_train[i])
        count = count + 1
        Y.append(YY[int(i), :])
    for i in num_bigwhole:
        XU1.append(list(Yy1[i, :]))
        xU1.append(list(x_train[i, :]))
        NUM1.append(y_train[i])
        count2 = count2 + 1
        Y1.append(YY[int(i), :])

    X_train_reduction1 = tf.nn.crelu(XU, name=None)  # CRELU
    X_train_reduction1 = tf.Session().run(X_train_reduction1)
    X_train_std = np.std(X_train_reduction1, axis=0)
    X_HE = X_train_reduction1 / X_train_std
    T = np.hstack((X_HE, xU))
    # print(np.array(T).shape)
    return T, XU, xU, Y, count, NUM, XU1, xU1, Y1, count2, NUM1, num_smallwhole, num_bigwhole




xunhuan = 0

sfolder = StratifiedKFold(n_splits=10, shuffle=True, random_state=5)
B = 0
C = 0
D = 0
E = 0
F = 0
G = 0
I = 0
J = 0
K = 0
L = 0
M = 0
N = 0
LL = 0
KK = 0
MM = 0
NN = 0
for train_index, valid_index in sfolder.split(X, Y):
    wstrat = time.time()
    x_train, x_valid = X[train_index], X[valid_index]

    y_train, y_valid = Y[train_index], Y[valid_index]

    a = 0
    righttrain = 0
    righttest = 0
    while a < 5:
        a = a + 1
        print(a)
        if a == 1:
            Sy = []
            for i in list(y_train):
                Sy.append(r1[int(i), :])


            YY = np.array(Sy)
            ST = []
            for i in list(y_valid):
                ST.append(r1[int(i), :])

            yy = np.array(ST)


            pca = PCA(n_components=zhuchengfen)
            print(x_train.shape)
            pca.fit(x_train)  # train_new)  # 用来训练模型
            X_train_reduction = pca.transform(x_train)  # train_new)  # 将数据集降维
            X_test_reduction = pca.transform(x_valid)  # test_new)
            X_train_reduction1 = preprocessing.scale(X_train_reduction)
            X_test_reduction1 = preprocessing.scale(X_test_reduction)
            X_train_std = np.std(X_train_reduction1, axis=0)
            X_test_std = np.std(X_test_reduction1, axis=0)
            X_train_reduction = X_train_reduction1 / X_train_std
            X_test_reduction = X_test_reduction1 / X_test_std
            X_train_reduction = tf.nn.crelu(X_train_reduction, name=None)
            X_test_reduction = tf.nn.crelu(X_test_reduction, name=None)
            X_train_reduction = sess.run(X_train_reduction)
            X_test_reduction = sess.run(X_test_reduction)

            ####


        elif a == 2:
            YY = lablesy
            pca = PCA(n_components=zhuchengfen)
            pca.fit(x_train)  # train_new)  # 用来训练模型
            X_train_reduction0 = pca.transform(x_train)  # train_new)  # 将数据集降维
            X_test_reduction0 = pca.transform(x_valid)  # test_new)
            X_train_reduction1 = preprocessing.scale(X_train_reduction0)
            X_test_reduction1 = preprocessing.scale(X_test_reduction0)
            X_train_std = np.std(X_train_reduction1, axis=0)
            X_test_std = np.std(X_test_reduction1, axis=0)
            X_train_reduction0 = X_train_reduction1 / X_train_std
            X_test_reduction0 = X_test_reduction1 / X_test_std
            X_train_reduction0 = tf.nn.crelu(X_train_reduction0, name=None)
            X_test_reduction0 = tf.nn.crelu(X_test_reduction0, name=None)
            X_train_reduction0 = sess.run(X_train_reduction0)
            X_test_reduction0 = sess.run(X_test_reduction0)
            X_test_reduction = []
            X_train_reduction = []
            for i in y1:
                X_train_reduction.append((X_train_reduction0[i]))
            for i in t1:
                X_test_reduction.append((X_test_reduction0[i]))
            X_train_reduction = np.array(X_train_reduction)
            X_test_reduction = np.array(X_test_reduction)



        elif a == 3:
            YY = lablesy3
            pca = PCA(n_components=zhuchengfen)
            pca.fit(SF0)  # train_new)  # 用来训练模型
            X_train_reduction0 = pca.transform(SF0)  # train_new)  # 将数据集降维
            X_test_reduction0 = pca.transform(ST0)  # test_new)
            X_train_reduction1 = preprocessing.scale(X_train_reduction0)
            X_test_reduction1 = preprocessing.scale(X_test_reduction0)
            X_train_std = np.std(X_train_reduction1, axis=0)
            X_test_std = np.std(X_test_reduction1, axis=0)
            X_train_reduction0 = X_train_reduction1 / X_train_std
            X_test_reduction0 = X_test_reduction1 / X_test_std
            X_train_reduction0 = tf.nn.crelu(X_train_reduction0, name=None)
            X_test_reduction0 = tf.nn.crelu(X_test_reduction0, name=None)
            X_train_reduction0 = sess.run(X_train_reduction0)
            X_test_reduction0 = sess.run(X_test_reduction0)
            X_test_reduction = []
            X_train_reduction = []
            for i in y13:
                X_train_reduction.append((X_train_reduction0[i]))
            for i in t13:
                X_test_reduction.append((X_test_reduction0[i]))
            X_train_reduction = np.array(X_train_reduction)
            X_test_reduction = np.array(X_test_reduction)



        elif a == 4:
            YY = lablesy5
            pca = PCA(n_components=zhuchengfen)
            pca.fit(SF2)  # train_new)  # 用来训练模型
            X_train_reduction0 = pca.transform(SF2)  # train_new)  # 将数据集降维
            X_test_reduction0 = pca.transform(ST2)  # test_new)
            X_train_reduction1 = preprocessing.scale(X_train_reduction0)
            X_test_reduction1 = preprocessing.scale(X_test_reduction0)
            X_train_std = np.std(X_train_reduction1, axis=0)
            X_test_std = np.std(X_test_reduction1, axis=0)
            X_train_reduction0 = X_train_reduction1 / X_train_std
            X_test_reduction0 = X_test_reduction1 / X_test_std
            X_train_reduction0 = tf.nn.crelu(X_train_reduction0, name=None)
            X_test_reduction0 = tf.nn.crelu(X_test_reduction0, name=None)
            X_train_reduction0 = sess.run(X_train_reduction0)
            X_test_reduction0 = sess.run(X_test_reduction0)
            X_test_reduction = []
            X_train_reduction = []
            for i in y15:
                X_train_reduction.append((X_train_reduction0[i]))
            for i in t15:
                X_test_reduction.append((X_test_reduction0[i]))
            X_train_reduction = np.array(X_train_reduction)
            X_test_reduction = np.array(X_test_reduction)

        else:
            YY = lablesy6
            pca = PCA(n_components=zhuchengfen)
            pca.fit(SF4)  # train_new)  # 用来训练模型
            X_train_reduction0 = pca.transform(SF4)  # train_new)  # 将数据集降维
            X_test_reduction0 = pca.transform(ST4)  # test_new)
            X_train_reduction1 = preprocessing.scale(X_train_reduction0)
            X_test_reduction1 = preprocessing.scale(X_test_reduction0)
            X_train_std = np.std(X_train_reduction1, axis=0)
            X_test_std = np.std(X_test_reduction1, axis=0)
            X_train_reduction0 = X_train_reduction1 / X_train_std
            X_test_reduction0 = X_test_reduction1 / X_test_std
            X_train_reduction0 = tf.nn.crelu(X_train_reduction0, name=None)
            X_test_reduction0 = tf.nn.crelu(X_test_reduction0, name=None)
            X_train_reduction0 = sess.run(X_train_reduction0)
            X_test_reduction0 = sess.run(X_test_reduction0)
            X_test_reduction = []
            X_train_reduction = []
            for i in y16:
                X_train_reduction.append((X_train_reduction0[i]))
            for i in t16:
                X_test_reduction.append((X_test_reduction0[i]))
            X_train_reduction = np.array(X_train_reduction)
            X_test_reduction = np.array(X_test_reduction)


        pianzhi_train = np.ones(X_train_reduction.shape[0]).reshape(-1, 1)
        pianzhi_test = np.ones(X_test_reduction.shape[0]).reshape(-1, 1)
        X_train_reduction1 = np.hstack((X_train_reduction, pianzhi_train))
        X_test_reduction1 = np.hstack((X_test_reduction, pianzhi_test))

        ###
        trainzhuanzhi = np.transpose(
            X_train_reduction1)
        YYY = []
        Yyy = []
        sum = 0
        n = 0

        YYY = YY

        YYY = np.array(YYY)

        YYY = tanhni(YYY)
        YyY = tanhdao(YYY)
        YYY = np.transpose(YYY)
        YyY = np.transpose(YyY)
        W1 = []
        quan = []

        zhuchengfen1 = zhuchengfen * 2


        for i in range(latent_variable_dim):
            H = np.tile(YyY[i, :], [zhuchengfen1 + 1,
                                    1])
            g2 = YyY[i, :] * YyY[i, :]
            yg2 = YYY[i, :] * g2  #
            yg2X = np.dot(yg2,
                          X_train_reduction1)
            Xnew = H * trainzhuanzhi


            XnewXnewT = np.dot(Xnew, np.transpose(Xnew))
            XnewXnewTni = np.linalg.inv(XnewXnewT)

            w1 = np.dot(yg2X, XnewXnewTni)

            W1.append(w1)
        W1 = np.array(W1)

        W1XUN = W1

        Yys = np.dot(W1XUN, np.transpose(
            X_train_reduction1))
        Tys = np.dot(W1XUN, np.transpose(
            X_test_reduction1))

        Yy = tanh(Yys)
        Ty = tanh(Tys)

        Yy2 = np.transpose(
            Yy)
        Ty2 = np.transpose(
            Ty)








        if a == 1:

            r11, r33 = gaosifenbuglass.gaosi_chubu(Yy2, len(list(y_train)), list(y_train))
            Yy1, FY0, SF0, lablesy, count, NUM, FY1, SF1, DAY1, cou1, num1, y1, y2 = threshold(Yy2, YY, x_train, y_train,
                                                                                           r33)
            Ty1, FT0, ST0, lamblest, count1, NUM1, FT1, ST1, DAY2, cou2, num2, t1, t2 = threshold(Ty2, yy, x_valid, y_valid,
                                                                                              r33)



            Yy_hang = gaosifenbuglass.gaosizhengtaifenbu(Yy2, r33, r11)
            Ty_hang = gaosifenbuglass.gaosizhengtaifenbu(Ty2, r33, r11)
            lablesy = np.array(lablesy)
            lamblest = np.array(lamblest)
            DAY1 = np.array(DAY1)
            DAY2 = np.array(DAY2)
            Tyh1 = []
            Yyh1 = []
            Tyh2 = []
            Yyh2 = []
            for i in y1:
                Yyh1.append(Yy_hang[i])

            for i in y2:
                Yyh2.append(Yy_hang[i])

            for i in t1:
                Tyh1.append(Ty_hang[i])
            for i in t2:
                Tyh2.append(Ty_hang[i])

            accuracy_train1 = np.sum(np.equal(Yyh1, NUM[:len(
                NUM)]))
            accuracy_test1 = np.sum(np.equal(Tyh1, NUM1[:len(
                NUM1)]))
            accuracy_train2 = np.sum(np.equal(Yyh2, num1[:len(num1)]))
            accuracy_test2 = np.sum(np.equal(Tyh2, num2[:len(num2)]))

            accuracy_train = np.sum(np.equal(Yy_hang, y_train[:len(y_train)]
                                             ))
            accuracy_test = np.sum(np.equal(Ty_hang, y_valid[:len(y_valid)]))
            B += accuracy_test / len(y_valid)

            print(B)



        elif a == 2:

            FY0 = np.array(FY0)
            SF0 = np.array(SF0)
            FT0 = np.array(FT0)
            ST0 = np.array(ST0)
            r11, r33 = gaosifenbuglass.gaosi_chubu(Yy2, count, NUM)  # 生成的是初级的高斯分布均值，方差，类数量
            Yy13, FY2, SF2, lablesy3, count3, NUM3, FY3, SF3, DAY13, cou13, num13, y13, y23 = threshold(Yy2, YY, SF0, NUM,
                                                                                                    r33)  # 新样本合并，满足小于的样本PEDCC,小于阈值的类标签，大于阈值的h 大于阈值的样本数, 大于阈值的类标签
            Ty13, FT2, ST2, lamblest3, count13, NUM13, FT3, ST3, DAY23, cou23, num23, t13, t23 = threshold(Ty2, yy, ST0,
                                                                                                       NUM1, r33)

            Yy_hang3 = gaosifenbuglass.gaosizhengtaifenbu(Yy2, r33, r11)
            Ty_hang3 = gaosifenbuglass.gaosizhengtaifenbu(Ty2, r33, r11)
            lablesy3 = np.array(lablesy3)
            Tyh13 = []
            Yyh13 = []
            Tyh23 = []
            Yyh23 = []
            for i in y13:
                Yyh13.append(Yy_hang3[i])

            for i in y23:
                Yyh23.append(Yy_hang3[i])

            for i in t13:
                Tyh13.append(Ty_hang3[i])
            for i in t23:
                Tyh23.append(Ty_hang3[i])

            accuracy_train5 = np.sum(np.equal(Yyh13, NUM3[:len(
                NUM3)]))
            accuracy_test5 = np.sum(np.equal(Tyh13, NUM13[:len(
                NUM13)]))
            accuracy_train6 = np.sum(np.equal(Yyh23, num13[:len(num13)]))
            accuracy_test6 = np.sum(np.equal(Tyh23, num23[:len(num23)]))

            accuracy_train7 = np.sum(np.equal(Yy_hang3, NUM[:len(NUM)]
                                              ))
            accuracy_test7 = np.sum(np.equal(Ty_hang3, NUM1[:len(NUM1)]))


            KK += (accuracy_test7 + accuracy_test2) / len(
                y_valid)




        elif a == 3:

            FY2 = np.array(FY2)
            SF2 = np.array(SF2)
            FT2 = np.array(FT2)
            ST2 = np.array(ST2)
            r11, r33 = gaosifenbuglass.gaosi_chubu(Yy2, count3, NUM3)  # 生成的是初级的高斯分布均值，方差，类数量
            Yy15, FY4, SF4, lablesy5, count5, NUM5, FY5, SF5, DAY15, cou15, num15, y15, y25 = threshold(Yy2, YY, SF2, NUM3,
                                                                                                    r33)

            Ty15, FT4, ST4, lamblest5, count15, NUM15, FT5, ST5, YY25, cou25, num25, t15, t25 = threshold(Ty2, yy, ST2,
                                                                                                      NUM13,
                                                                                                      r33)
            lablesy5 = np.array(lablesy5)
            lamblest5 = np.array(lamblest5)

            Yy_hang5 = gaosifenbuglass.gaosizhengtaifenbu(Yy2, r33, r11)
            Ty_hang5 = gaosifenbuglass.gaosizhengtaifenbu(Ty2, r33, r11)

            Tyh15 = []
            Yyh15 = []
            Tyh25 = []
            Yyh25 = []
            for i in y15:
                Yyh15.append(Yy_hang5[i])

            for i in y25:
                Yyh25.append(Yy_hang5[i])

            for i in t15:
                Tyh15.append(Ty_hang5[i])
            for i in t25:
                Tyh25.append(Ty_hang5[i])

            accuracy_train13 = np.sum(np.equal(Yyh15, NUM5[:len(
                NUM5)]))
            accuracy_test13 = np.sum(np.equal(Tyh15, NUM15[:len(
                NUM15)]))
            accuracy_train14 = np.sum(np.equal(Yyh25, num15[:len(num15)]))
            accuracy_test14 = np.sum(np.equal(Tyh25, num25[:len(num25)]))

            accuracy_train15 = np.sum(np.equal(Yy_hang5, NUM3[:len(NUM3)]
                                               ))
            accuracy_test15 = np.sum(np.equal(Ty_hang5, NUM13[:len(NUM13)]))


            LL += (accuracy_test15 + +accuracy_test6 + accuracy_test2) / len(
                y_valid)





        elif a == 4:

            r11, r33 = gaosifenbuglass.gaosi_chubu(Yy2, count5, NUM5)  # 生成的是初级的高斯分布均值，方差，类数量
            FY4 = np.array(FY4)
            SF4 = np.array(SF4)
            FT4 = np.array(FT4)
            ST4 = np.array(ST4)

            Yy16, FY6, SF6, lablesy6, count6, NUM6, FY7, SF7, DAY16, cou16, num16, y16, y26 = threshold(Yy2, YY, SF4, NUM5,
                                                                                                    r33)
            Ty16, FT6, ST6, lamblest6, count16, NUM16, FT7, ST7, DAY26, cou26, num26, t16, t26 = threshold(Ty2, yy, ST4,
                                                                                                       NUM15, r33)

            Yy_hang6 = gaosifenbuglass.gaosizhengtaifenbu(Yy2, r33, r11)
            Ty_hang6 = gaosifenbuglass.gaosizhengtaifenbu(Ty2, r33, r11)
            lablesy6 = np.array(lablesy6)
            lamblest6 = np.array(lamblest6)
            Tyh16 = []
            Yyh16 = []
            Tyh26 = []
            Yyh26 = []
            for i in y16:
                Yyh16.append(Yy_hang6[i])

            for i in y26:
                Yyh26.append(Yy_hang6[i])

            for i in t16:
                Tyh16.append(Ty_hang6[i])
            for i in t26:
                Tyh26.append(Ty_hang6[i])

            accuracy_train00 = np.sum(np.equal(Yyh16, NUM6[:len(
                NUM6)]))

            accuracy_test00 = np.sum(np.equal(Tyh16, NUM16[:len(
                NUM16)]))
            accuracy_train01 = np.sum(np.equal(Yyh26, num16[:len(num16)]))
            accuracy_test01 = np.sum(np.equal(Tyh26, num26[:len(num26)]))

            accuracy_train02 = np.sum(np.equal(Yy_hang6, NUM5[:len(NUM5)]
                                               ))
            accuracy_test02 = np.sum(np.equal(Ty_hang6, NUM15[:len(NUM15)]))

            MM += (accuracy_test02 + accuracy_test14 + +accuracy_test6 + accuracy_test2) / len(
                y_valid)


        else:

            r11, r33 = gaosifenbuglass.gaosi_chubu(Yy2, count6, NUM6)  # 生成的是初级的高斯分布均值，方差，类数量

            Ty_hang = gaosifenbuglass.gaosizhengtaifenbu(Ty2, r33, r11)
            Yy_hang = gaosifenbuglass.gaosizhengtaifenbu(Yy2, r33, r11)

            accuracy_train000 = np.sum(np.equal(Yy_hang, NUM6[:len(
                NUM6)]))
            accuracy_test000 = np.sum(np.equal(Ty_hang, NUM16[:len(NUM16)]))  # / count1  #
            NN += (accuracy_test000 + accuracy_test01 + accuracy_test14 + +accuracy_test6 + accuracy_test2) / len(
                y_valid)
            print(NN)

        righttest = righttest + accuracy_test
        righttrain = righttrain + accuracy_train

    accuracy_test = righttest / len(y_valid)
    accuracy_train = righttrain / len(y_train)
    sum1 = sum1 + accuracy_train
    sum2 = sum2 + accuracy_test

print(B / 10)
print(KK / 10)
print(LL / 10)
print(MM / 10)
print(NN / 10)

