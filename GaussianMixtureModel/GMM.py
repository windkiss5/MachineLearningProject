#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : GMM.py
# @Author: WINDKISS
# @Date  : 2020/11/24
# @Function  : 高斯混合模型GMM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class GMM():
    def __init__(self, X, K):
        self.X = X
        self.K = K
        # sampleSize：训练集数量    features：样本特征数目
        self.sampleSize, self.features = np.shape(self.X)
        # alpha：各隐变量先验概率
        self.alpha = np.random.dirichlet(np.ones(K), size=1)[0].reshape(K, 1)
        # mu：各高斯分模型的均值
        self.mu = np.zeros((self.K, self.features))
        # sigma：各高斯分模型的协方差矩阵
        self.sigma = np.random.normal(loc=5.0, scale=1.5, size=(self.K, self.features, self.features))
        # 各Xi在当前各seta下的概率矩阵
        self.GaussianMatrix = np.zeros((self.sampleSize, self.K))
        # 隐变量的后验概率
        self.ZMatrix = np.zeros((self.sampleSize, self.K))
        # 预测矩阵
        self.preMatrix = np.zeros((self.sampleSize, self.K))
        # 聚类结果
        self.prediction = np.ones((self.sampleSize, 1)) * -1

    def calGaussian(self, x, mu, sigma):
        """
        计算对应mu, sigma的高斯模型下x的概率值
        :param x: 样本
        :param mu: 均值
        :param sigma: 协方差矩阵
        :return: 概率值
        """
        # reshape
        x = np.reshape(x, (self.features, 1))
        mu = np.reshape(mu, (self.features, 1))
        sigma = np.reshape(sigma, (self.features, self.features))

        # |sigma|
        detSigma = np.abs(np.linalg.det(sigma))
        if detSigma - 0. < 0.000001:
            detSigma += 0.001
        # ------------- calGaussian ------------- #
        left = 1 / (np.power(2 * np.pi, self.features / 2) * np.power(detSigma, 1 / 2))
        # noinspection PyBroadException
        try:
            tmp = (x - mu).T.dot(np.linalg.inv(sigma)).dot(x - mu)
        except:
            tmp = (x - mu).T.dot(np.linalg.pinv(sigma)).dot(x - mu)
        right = np.exp(-1 / 2 * tmp)
        val = left * right

        return val

    def GMM_EM(self, iter=100):
        """
        EM算法求解GMM
        :param iter: 最大迭代次数
        :return: 聚类结果
        """
        # iterStep：当前迭代次数，超过设置次数还未收敛则强制停止
        iterStep = 0
        # paramChanged：单次迭代中有参数改变则增加1
        paramChanged = 1
        # --------------- EM Algorithm ---------------- #
        while iterStep < iter:
            iterStep += 1
            print("------- iterStep = {} -------".format(iterStep))
            # 计算各Xi在当前各seta下的概率矩阵
            for i in range(self.sampleSize):
                for k in range(self.K):
                    self.GaussianMatrix[i][k] = self.calGaussian(self.X[i], self.mu[k], self.sigma[k])
            # 计算P(z^i = z_k|x_i, seta^t)
            for i in range(self.sampleSize):
                tmp = np.reshape(self.GaussianMatrix[i], (1, self.K)).dot(self.alpha)[0]
                for k in range(self.K):
                    self.ZMatrix[i][k] = self.alpha[k] * self.GaussianMatrix[i, k] / tmp

            # alpha
            for k in range(self.K):
                self.alpha[k] = self.ZMatrix[k, :].sum() / self.sampleSize

            # mu
            for k in range(self.K):
                tmp = []
                for i in range(self.sampleSize):
                    tmp = self.X[i] * self.ZMatrix[i, k]
                self.mu[k] = tmp / self.ZMatrix[k].sum()

            # sigma
            for k in range(self.K):
                tmp = []
                for i in range(self.sampleSize):
                    tmp = self.X[i].T.dot(self.X[i]) * self.ZMatrix[i, k]
                self.sigma[k] = tmp / self.ZMatrix[k].sum()

        # --------------- 聚类 ---------------- #
        maxClass = -1
        for i in range(self.sampleSize):
            for k in range(self.K):
                self.preMatrix[i][k] = self.calGaussian(self.X[i], self.mu[k], self.sigma[k])
            self.prediction[i] = np.argmax(self.preMatrix[i, :])
            if self.prediction[i] > maxClass:
                maxClass = self.prediction[i]
        return self.prediction, maxClass


def main():
    # --------------- 生成数据 ---------------- #
    num = 100
    # 类别1
    X1 = np.random.normal(loc=2.2, scale=1.0, size=(num, 2))
    # 类别2
    X2 = np.random.normal(loc=6.5, scale=0.4, size=(num, 2))
    # 类别3
    X3 = np.random.normal(loc=12.5, scale=1.5, size=(num * 2, 2))
    # 组合
    X = np.concatenate((X1, X2, X3), axis=0)
    plt.plot(X1[:, 0], X1[:, 1], "o")
    plt.plot(X2[:, 0], X2[:, 1], "o")
    plt.plot(X3[:, 0], X3[:, 1], "o")
    plt.show()

    # --------------- GMM ---------------- #
    gmm = GMM(X, 3)
    pre, maxClass = gmm.GMM_EM()
    print(pd.DataFrame(pre))
    print(maxClass)
    # --------------- Plot ---------------- #
    for i in range(num * 4):
        if pre[i] == 0:
            plt.plot(X[i][0], X[i][1], 'bo')
        elif pre[i] == 1:
            plt.plot(X[i][0], X[i][1], 'ro')
        elif pre[i] == 2:
            plt.plot(X[i][0], X[i][1], 'go')
    plt.show()


if __name__ == '__main__':
    main()
