# -*- coding:utf-8 -*-
# @Time : 2020/8/31 19:26
# @Author : WINDKISS
# @File : LDA.py
# @Function : 线性判别分析，也叫fisher线性判别分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class LDA:
    def __init__(self, X, Y):
        self.X = np.c_[np.ones(len(X)), X]
        self.Y = Y
        self.W = np.random.randn(self.X.shape[1])

    def LDAsolver(self):
        # 拆分样本
        # 类别1
        index1 = np.where(self.Y == 1)
        X1 = self.X[index1[0]]
        Y1 = self.Y[index1[0]]
        # 类别2
        index2 = np.where(self.Y == -1)
        X2 = self.X[index2[0]]
        Y2 = self.Y[index2[0]]
        # 计算均值
        meanX1 = np.mean(X1, axis=0)
        meanX2 = np.mean(X2, axis=0)
        # 计算方差
        covX1 = np.cov(X1, rowvar=False)
        covX2 = np.cov(X2, rowvar=False)
        print(pd.DataFrame(covX1))
        print("==============X2=========")
        print(pd.DataFrame(covX2))
        # 计算W
        self.W = np.linalg.pinv((covX1 + covX2)).dot(meanX1 - meanX2)
        return self.W


def main():
    # 生成数据
    num = 100

    # 类别 1
    X1 = np.random.normal(loc=2.3, scale=0.7, size=(num, 2))
    print(pd.DataFrame(X1))
    print("=============")
    Y1 = -1 * np.ones((num, 1))
    # 类别 2
    X2 = np.random.normal(loc=-0.9, scale=0.5, size=(num, 2))
    Y2 = np.ones((num, 1))

    plt.figure()
    plt.plot(X1[:, 0], X1[:, 1], "*")
    plt.plot(X2[:, 0], X2[:, 1], "*")


    X = np.concatenate((X1, X2), axis=0)  # 拼接成一个整体
    Y = np.concatenate((Y1, Y2), axis=0)
    # 打乱次序
    index = np.random.permutation(X.shape[0])
    X = X[index]
    Y = Y[index]

    W = LDA(X, Y).LDAsolver()
    print("===================")
    print(pd.DataFrame(W))

    Xx = np.linspace(-1, 3, 50)
    print(Xx)
    Y_x = (-1 * W[1] / W[2]) * Xx - W[0] / W[2]
    print(Y_x)
    plt.plot(Xx, Y_x)
    plt.show()

if __name__ == '__main__':
    main()