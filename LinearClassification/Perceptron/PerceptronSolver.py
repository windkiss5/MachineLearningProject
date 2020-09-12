# -*- coding:utf-8 -*-
# @Time : 2020/8/29 15:27
# @Author : WINDKISS
# @File : PerceptronSolver.py
# @Function : 感知机算法
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PerceptronSolver:
    def __init__(self, X, Y, epoch=1000, learningRate=0.01, batchSize = 10):
        self.X = np.c_[np.ones(len(X)), X]
        self.Y = Y
        self.epoch = epoch
        self.learningRate = learningRate
        self.W = np.random.randn(self.X.shape[1])
        self.sampleSize = self.X.shape[0]
        self.batchSize = batchSize
        # 计算每轮要传多少条数据
        self.step = self.sampleSize // self.batchSize

    def loadData(self):
        # 打乱训练集
        index = np.random.permutation(self.sampleSize)
        Xshullfled = self.X[index]
        Yshullfled = self.Y[index]
        for i in range(self.step):
            # 每次读取batchSize份数据
            Xbatch = Xshullfled[i * self.batchSize: (i + 1) * self.batchSize]
            Ybatch = Yshullfled[i * self.batchSize: (i + 1) * self.batchSize]
            yield Xbatch, Ybatch

    @staticmethod
    def sign(Y):
        Y[Y >= 0] = 1
        Y[Y < 0] = -1
        return Y

    def perceptronClassify(self):
        for i in range(self.epoch):
            print("========第", i, "轮=========")
            p = 1
            for Xbatch, Ybatch in self.loadData():
                p += 1
                print("========第", p, "批次=========")
                # 计算Y预测值
                gradient = np.zeros(self.W.shape)
                YPredicted = PerceptronSolver.sign(Xbatch.dot(self.W)).T
                for j in range(len(YPredicted)):
                    if YPredicted[j] != Ybatch[j]:
                        # 错误分类
                        gradient -= Ybatch[j] * Xbatch[j]
                # 梯度下降
                self.W = self.W - self.learningRate * gradient
        return np.array(self.W)

    def perceptronClassifyNormal(self):
        for i in range(self.epoch):
            # 计算Y预测值
            gradient = np.zeros(self.W.shape)
            YPredicted = PerceptronSolver.sign(self.X.dot(self.W)).T
            for j in range(len(YPredicted)):
                if YPredicted[j] != self.Y[j]:
                    # 错误分类
                    gradient -= self.Y[j] * self.X[j]
            # 梯度下降
            self.W = self.W - self.learningRate * gradient
        return np.array(self.W)

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

    W = PerceptronSolver(X, Y).perceptronClassify()
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