# -*- coding:utf-8 -*-
# @Time : 2020/8/28 19:43
# @Author : WINDKISS
# @File : LinearRegressionSolver.py
# @Function : 线性回归 - 多维度 - 梯度下降法求解or公式法 - L2正则
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegressionSolver:
    def __init__(self, X, Y, regular=None, coefficient=0.):
        # 训练数据与标签
        # X = (x1 x2 x3 ... xN)T,其中xi为列向量
        # 注意！！！！在转置后X中每个xi(样本)都变成了行向量，而不是列向量
        #     [2 3 6 7 8 9 5]       [2]
        # X = [3 7 9 2 9 7 2]   Y = [7]
        #     [8 0 4 4 5 7 3]       [9]
        # 这里一共有3个样本而不是7个样本，而对于偏置b的计算，令b = w0*x00,且令x00=1，这里w0和x00都是实数，
        # x0是指在样本列向量的前面添1，而在X矩阵上的表现就是在X前面添加一列全1
        # 当然，若你令 b = wN+1 * xN+1N+1, 那么相应的，是需要在X末尾添一列全1，那么最后计算出来的W中b就是W列向量中最后面的那个数了！！！
        #     [1 2 3 6 7 8 9 5]       [2]
        # X = [1 3 7 9 2 9 7 2]   Y = [7]
        #     [1 8 0 4 4 5 7 3]       [9]
        self.X = np.c_[np.ones(len(X)), X]
        self.Y = Y
        self.sampleSize = len(Y)
        self.characteristicSize = self.X.ndim
        # 按高斯分布随机初始化W矩阵，W是一个列向量，且长度为每个训练样本特征数+1(X.ndim表示X的列数，此处X已经添了1)
        # self.W = np.array([3.66, 2.11]).T
        self.W = np.random.randn(self.characteristicSize)
        self.regular = regular
        self.coefficient = coefficient
        print(self.W, len(self.W), self.W.ndim)

    # 公式法求解 W = (X转置X)^(-1)X转置Y
    def formulaMethod(self):
        # 无正则
        if self.regular is None:
            self.W = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.Y)
        # 有正则
        else:
            self.W = np.linalg.inv(self.X.T.dot(self.X) + self.coefficient * np.identity(self.characteristicSize)).dot(self.X.T).dot(self.Y)
        return np.array(self.W)

    # 梯度下降法 W = W - α * ▽W
    def GradientDescentMethod(self, learningRate=0.0001, epoch=1000):
        for i in range(epoch):
            # 计算梯度
            # 无正则
            if self.regular is None:
                gradient = (2 / self.sampleSize) * (self.X.T.dot(self.X).dot(self.W) - self.X.T.dot(self.Y))
            # 有正则
            else:
                gradient = (2 / self.sampleSize) * ((self.X.T.dot(self.X) + self.coefficient * np.identity(self.characteristicSize)).dot(self.W) - self.X.T.dot(self.Y))
            # 更新梯度
            self.W = self.W - learningRate * gradient
            print(self.W)
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

    W = LinearRegressionSolver(X, Y).formulaMethod()
    print("===================")
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