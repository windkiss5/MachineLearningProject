# Python
# -*- coding: UTF-8 -*-
# @Time: 2020/9/6 21:56
# @Author: WINDKISS
# @File: GDA.py
# @Function: 高斯判别分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GDA():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.samplesize = self.X.shape[0]
        self.theta = 0
        self.mu1 = np.zeros((self.X.shape[1])).reshape(self.X.shape[1], 1)
        self.mu2 = np.zeros((self.X.shape[1])).reshape(self.X.shape[1], 1)
        self.sigma = np.zeros((self.X.shape[1], self.X.shape[1])).reshape(self.X.shape[1], self.X.shape[1])

    def GDAFormulaMethod(self):
        # 拆分样本
        # 类别1
        index1 = np.where(self.Y == 1)
        X1 = self.X[index1[0]]
        N1 = X1.shape[0]
        Y1 = self.Y[index1[0]]
        # 类别2
        index2 = np.where(self.Y == 0)
        X2 = self.X[index2[0]]
        N2 = X2.shape[0]
        Y2 = self.Y[index2[0]]

        # 计算theta
        self.theta = X1.shape[0] / self.samplesize
        # 计算mu1, mu2
        for i in range(self.samplesize):
            self.mu1 += (self.Y[i] * self.X[i]).reshape(self.X.shape[1], 1)
            self.mu2 += ((1 - self.Y[i]) * self.X[i]).reshape(self.X.shape[1], 1)
        self.mu1 /= N1
        self.mu2 /= N2
        # 计算sigma
        # 计算方差
        S1 = np.cov(X1, rowvar=False)
        S2 = np.cov(X2, rowvar=False)
        self.sigma = (N1 * S1 + N2 * S2) / self.samplesize

        return self.theta, self.mu1, self.mu2, self.sigma

    @staticmethod
    # 预测单个数据
    def predict(X, theta, mu1, mu2, sigma):
        detSigma = np.linalg.det(sigma)
        features = X.shape[0]
        P1 = np.log(1 / np.power(2* np.pi, features / 2) * np.power(detSigma, 1/2)) -\
             (1/2)*(X - mu1).T.dot(np.linalg.inv(sigma)).dot(X - mu1) + \
             np.log(theta)
        P2 = np.log(1 / np.power(2 * np.pi, features / 2) * np.power(detSigma, 1 / 2)) - \
             (1 / 2) * (X - mu2).T.dot(np.linalg.inv(sigma)).dot(X - mu2) + \
             np.log(1 - theta)

        if P1 > P2:
            return 1
        else:
            return 0


def main():
    # 生成数据
    num = 100

    # 类别 1
    X1 = np.random.normal(loc=2.3, scale=0.7, size=(num, 2))
    Y1 = np.ones((num, 1))
    # 类别 2
    X2 = np.random.normal(loc=-0.9, scale=0.5, size=(num, 2))
    Y2 = 0 * np.ones((num, 1))


    plt.plot(X1[:, 0], X1[:, 1], "*")
    plt.plot(X2[:, 0], X2[:, 1], "*")

    X = np.concatenate((X1, X2), axis=0)  # 拼接成一个整体
    Y = np.concatenate((Y1, Y2), axis=0)
    # 打乱次序
    index = np.random.permutation(X.shape[0])
    X = X[index]
    Y = Y[index]

    theta, mu1, mu2, sigma  = GDA(X, Y).GDAFormulaMethod()

    # 预测
    # 类别 1
    XP1 = np.random.normal(loc=2.3, scale=0.7, size=(2, 1))
    print("数据:(",XP1[0][0],",",XP1[1][0],")","事实分类为：第1类，预测结果为：第",GDA.predict(XP1, theta, mu1, mu2, sigma),"类")
    plt.plot(XP1[0][0], X1[1][0], "^", markersize=15)

    XP2 = np.random.normal(loc=-0.9, scale=0.5, size=(2, 1))
    print("数据:(",XP2[0][0],",",XP2[1][0],")","事实分类为：第0类，预测结果为：第",GDA.predict(XP2, theta, mu1, mu2, sigma),"类")
    plt.plot(XP2[0][0], X2[1][0], "o", markersize=15)

    plt.show()

if __name__ == '__main__':
    main()
