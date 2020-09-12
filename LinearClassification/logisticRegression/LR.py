# Python
# -*- coding: UTF-8 -*-
# @Time: 2020/9/2 22:30
# @Author: WINDKISS
# @File: LR.py
# @Function: 软输出-概率判别模型-逻辑回归
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LR:
    def __init__(self, X, Y, epoch=1000, lr=0.001):
        self.X = np.c_[np.ones(len(X)), X]
        self.Y = Y
        self.epoch = epoch
        self.lr = lr
        self.W = np.random.randn(self.X.shape[1]).reshape(self.X.shape[1], 1)
        self.sampleSize = self.X.shape[0]
        self.features = self.X.shape[1]

    @staticmethod
    def sigmod(Z):
        return 1/(1 + np.exp(-Z))

    def LRNormal(self):
        for i in range(self.epoch):
            # 计算预测值
            YPredicted = LR.sigmod(self.X.dot(self.W)).reshape((self.sampleSize, 1))
            # 计算梯度
            gradient = self.X.T.dot(self.Y - YPredicted)
            # 梯度下降
            self.W -= self.lr * gradient
        return self.W

def main():
    # 生成数据
    num = 100

    # 类别 1
    X1 = np.random.normal(loc=2.3, scale=0.7, size=(num, 2))
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

    W = LR(X, Y).LRNormal()
    print(pd.DataFrame(W))
#
    Xx = np.linspace(-1, 3, 50)
    print(Xx)
    Y_x = (-1 * W[1] / W[2]) * Xx - W[0] / W[2]
    print(Y_x)
    plt.plot(Xx, Y_x)
    plt.show()

if __name__ == '__main__':
    main()