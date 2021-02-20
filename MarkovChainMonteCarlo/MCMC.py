#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : MCMC.py
# @Author: WINDKISS
# @Date  : 2021/02/20
# @Function  : 马尔可夫链蒙特卡洛(MCMC) - MCMC采样 - M-H采样 - Gibbs采样

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd


class MCMC:
    """
    实例中待采样分布将直接采用连续型分布，离散分布不再讨论。
    """

    def __init__(self, n1, n2):
        """
        :param n1: 燃烧期迭代次数
        :param n2: 需要采样的样本数量
        """
        self.n1 = n1
        self.n2 = n2
        self.sigma = 1.0
        # Beta分布的参数
        self.a = 2.0
        self.b = 6.0

    def Beta(self, x):
        return stats.beta(self.a, self.b).pdf(x)

    def Norm(self, loc, x):
        return stats.norm.pdf(x, loc, self.sigma)

    def MCMCAlgo(self):
        """
        待采样分布设定为一维Beta分布(概率密度值好求，但很难逆求CDF)
        转移核函数采用方差为1，均值为前一样本的条件概率分布
        """
        X = [0 for _ in range(self.n2)]

        "1. 随机生成初始状态 - scale为标准差"
        x_pre = np.random.normal(loc=0, scale=self.sigma, size=1)
        for i in range(self.n1 + self.n2):
            "a. 根据x_pre在转移核函数中采样候选样本x_"
            # x_ = np.random.normal(loc=x_pre, scale=self.sigma, size=1)[0]
            x_ = stats.norm.rvs(loc=x_pre, scale=self.sigma, size=1, random_state=None)[0]
            print("------- MCMCAlgo iterStep = {} -------".format(i))
            "b. 计算接收率"
            alpha = self.Beta(x_) * self.Norm(x_pre, x_)
            "c. 从均匀分布中采样得到mu"
            mu = np.random.uniform(0, 1)
            "d. 判断是否接收样本"
            if mu < alpha:
                x_pre = x_
                if i >= self.n1:
                    X[i - self.n1] = x_
            else:
                if i >= self.n1:
                    X[i - self.n1] = x_pre
        "打印图表"
        # 打印Beta(2, 6)
        x = np.linspace(0, 1, 1002)[1:-1]
        dist = stats.beta(self.a, self.b)
        plt.plot(x, dist.pdf(x), c='tab:blue', label=r'$\alpha=%.1f,\ \beta=%.1f$' % (self.a, self.b))
        # 打印采样数据
        numBins = 100
        plt.hist(X, numBins, density=1, facecolor='red', alpha=0.7)
        plt.show()

    def Metroplis_hastings_Algo(self):
        """
        待采样分布设定为一维Beta分布(概率密度值好求，但很难逆求CDF)
        转移核函数采用方差为1，均值为前一样本的条件概率分布
        """
        X = [0 for _ in range(self.n2)]

        "1. 随机生成初始状态 - scale为标准差"
        x_pre = np.random.normal(loc=0, scale=self.sigma, size=1)
        for i in range(self.n1 + self.n2):
            "a. 根据x_pre在转移核函数中采样候选样本x_"
            # x_ = np.random.normal(loc=x_pre, scale=self.sigma, size=1)[0]
            x_ = stats.norm.rvs(loc=x_pre, scale=self.sigma, size=1, random_state=None)[0]
            print("------- Metroplis_hastings_Algo iterStep = {} -------".format(i))
            "b. 计算接收率 注：在选取的方差为1的N(x_pre, 1)的核函数下，Norm(x_pre, x_) = Norm(x_, x_pre)恒成立，但为了理解公式还是写明"
            # alpha = min(1., self.Beta(x_) / self.Beta(x_pre))
            alpha = min(1, (self.Beta(x_) * self.Norm(x_, x_pre)) / (self.Beta(x_pre) * self.Norm(x_pre, x_)))
            "c. 从均匀分布中采样得到mu"
            mu = np.random.uniform(0, 1)
            "d. 判断是否接收样本"
            if mu < alpha:
                x_pre = x_
                if i >= self.n1:
                    X[i - self.n1] = x_
            else:
                if i >= self.n1:
                    X[i - self.n1] = x_pre
        "打印图表"
        # 打印Beta(2, 6)
        x = np.linspace(0, 1, 1002)[1:-1]
        dist = stats.beta(self.a, self.b)
        plt.plot(x, dist.pdf(x), c='tab:blue', label=r'$\alpha=%.1f,\ \beta=%.1f$' % (self.a, self.b))
        # 打印采样数据
        numBins = 100
        plt.hist(X, numBins, density=1, facecolor='red', alpha=0.7)
        plt.show()

    def run(self):
        self.MCMCAlgo()
        self.Metroplis_hastings_Algo()


class Gibbs:
    """
    由于二维高斯分布的各维度对应的条件概率也是高斯分布，于是示例中的待采样分布设定为均值为[5, -1]T，协方差矩阵为[[1, 1], [1, 4]]的二维高斯分布

    那么可以得到两个维度[Xa, Xb]T的条件概率为：
    P(Xa|Xb) = N(mu_a.b + Σ_ab((Σ_bb)^-1)Xb, Σ_a.b)
    P(Xb|Xa) = N(mu_b.a + Σ_ba((Σ_aa)^-1)Xa, Σ_b.a)
    其中
    mu = [mu_a, mu_b]T
    Σ = [[Σ_aa, Σ_ab], [Σ_ba, Σ_bb]]
    mu_a.b = mu_a - Σ_ab((Σ_bb)^-1)mu_b
    mu_b.a = mu_b - Σ_ba((Σ_aa)^-1)mu_a
    Σ_a.b = Σ_aa - Σ_ab((Σ_bb)^-1)Σ_ba
    Σ_b.a = Σ_bb - Σ_ba((Σ_aa)^-1)Σ_ab

    具体推导见笔记3-2
    """

    def __init__(self, n1, n2):
        """
        :param n1: 燃烧期迭代次数
        :param n2: 需要采样的样本数量
        """
        self.n1 = n1
        self.n2 = n2
        # 待采样分布的均值与协方差矩阵
        self.mu = [5, -1]
        self.cov = [[1, 1], [1, 4]]
        # 中间变量
        self.mu_a_b = self.mu[0] - self.cov[0][1] / self.cov[1][1] * self.mu[1]
        self.mu_b_a = self.mu[1] - self.cov[1][0] / self.cov[0][0] * self.mu[0]
        self.sigma_a_b = self.cov[0][0] - self.cov[0][1] / self.cov[1][1] * self.cov[1][0]
        self.sigma_b_a = self.cov[1][1] - self.cov[1][0] / self.cov[0][0] * self.cov[0][1]

    def Norm(self, x):
        return stats.multivariate_normal.pdf(x, mean=self.mu, cov=self.cov)

    def GenerateNorm(self, X_, index):
        """
        采样
        :param X_: 上一个样本
        :param index: 标记哪个作为条件 0-Xa 1-Xb
        :return: 新采样的样本
        """
        if index == 0:
            mu_ = self.mu_b_a + self.cov[1][0] / self.cov[0][0] * X_
            cov_ = self.sigma_b_a
        else:
            mu_ = self.mu_a_b + self.cov[0][1] / self.cov[1][1] * X_
            cov_ = self.sigma_a_b
        return stats.norm.rvs(loc=mu_, scale=cov_, size=1, random_state=None)[0]

    def run(self):
        X = [0. for _ in range(self.n2)]
        Y = [0. for _ in range(self.n2)]
        P = [0. for _ in range(self.n2)]
        "1. 随机生成初始状态 - scale为标准差"
        Xa_pre = np.random.normal(loc=self.mu[1], scale=1, size=1)
        for i in range(self.n1 + self.n2):
            print("------- GibbsAlgo iterStep = {} -------".format(i))
            "a. 从条件概率P(Xb|Xa = Xa_pre)中采样得到Xb维度的样本Xb_"
            Xb_ = self.GenerateNorm(Xa_pre, 0)
            "b. 从条件概率P(Xa|Xb = Xb_)中采样得到Xa维度的样本Xa_"
            Xa_ = self.GenerateNorm(Xb_, 1)
            if i >= self.n1:
                X[i - self.n1] = Xa_
                Y[i - self.n1] = Xb_
                P[i - self.n1] = self.Norm([Xa_, Xb_])
            Xa_pre = Xa_

        "打印图表"
        # 各维度采样结果
        num_bins = 100
        plt.hist(X, num_bins, density=1, facecolor='green', alpha=0.5)
        plt.hist(Y, num_bins, density=1, facecolor='red', alpha=0.5)
        plt.title('Histogram')
        plt.show()
        # 3D图表
        fig = plt.figure()
        ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
        ax.scatter(X, Y, P, marker='o')
        plt.show()


if __name__ == '__main__':
    # mcmc = MCMC(50000, 30000)
    # mcmc.run()

    gibbs = Gibbs(50000, 30000)
    gibbs.run()
