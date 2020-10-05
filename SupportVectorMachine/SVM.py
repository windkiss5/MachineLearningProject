# Python
# -*- coding: UTF-8 -*-
# @Time: 2020/10/4 21:12
# @Author: WINDKISS
# @File: SVM.py
# @Function: What SVM.py can do?
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, X, Y, C=200, toler=0.001, sigma=10):
        """
        SVM类 本SVM仅实现了高斯核下对双月分类数据集的分类
        :param X: 训练集数据
        :param Y: 训练集标签 Y∈{-1， 1}
        :param C: 惩罚常数
        :param toler: 控制变量变化阈值
        :param sigma: 高斯核函数相关
        """
        self.X = X
        self.Y = Y
        self.W = None
        self.B = 0.
        self.C = C
        self.sigma = sigma
        self.toler = toler
        # simpleSize：训练集数量    features：样本特征数目
        self.simpleSize, self.features = np.shape(self.X)
        # lambdas: 待优化拉格朗日算子
        self.lambdas = [0.] * self.simpleSize
        # E: Ej = f(xj) - yj
        self.E = -1 * self.Y
        # 支持向量下标集
        self.supportVecIndex = []
        # 高斯核函数矩阵(RBF核)
        self.K = self.calcKernel()

    def calcKernel(self):
        """
        计算核函数矩阵
        :return: 高斯核函数矩阵
        """
        K = [[0. for _ in range(self.simpleSize)] for _ in range(self.simpleSize)]
        for i in range(self.simpleSize):
            # 取出下标为i的样本
            Xi = self.X[i, :]
            for j in range(i, self.simpleSize):
                Xj = self.X[j, :]
                # 先计算||Xi - Xj||^2
                result = (Xi - Xj) * (Xi - Xj).T
                # 分子除以分母后去指数，得到的即为高斯核结果
                result = np.exp(-1 * result / (2 * self.sigma ** 2))
                # 将Xi*Xj的结果存放入k[i][j]和k[j][i]中
                K[i][j] = result
                K[j][i] = result
        return K

    def calcFxj(self, j):
        """
        计算f(xj) = Σlambdai*Yi*Kij + B
        :param j: 样本下标
        :return: f(xj)
        """
        # 计算f(xi)的一个trick就是：由于分类正确的样本所对应的lambda恒为0，所以可以先筛选出lambda不为0的那些样本进行求和
        index = [i for i, lambdas in enumerate(self.lambdas) if lambdas != 0]
        result = 0.
        for i in index:
            result += self.lambdas[i] * self.Y[i] * self.K[i][j]
        result += self.B
        return result

    def isSatisfyKKT(self, i):
        """
        判断下标为i的样本是否符合KKT条件
        :param i: 下标
        :return: True or False
        """
        fxi = self.calcFxj(i)
        if math.fabs(self.lambdas[i]) <= self.toler and self.Y[i] * fxi >= 1:
            # 第一类： 分类正确
            return True
        elif -1 * self.toler <= self.lambdas[i] <= self.C + self.toler and self.Y[i] * fxi == 1:
            # 第二类： 支持向量
            return True
        elif math.fabs(self.lambdas[i] - self.C) <= self.toler and self.Y[i] * fxi <= 1:
            # 第三类： 分类错误
            return True
        return False

    def calcEiNew(self, j):
        """
        计算Ej = f(xj) - yj
        :param j: 样本下标
        :return: Ej
        """
        return self.calcFxj(j) - self.Y[j]

    def searchLambda2(self, i):
        """
        启发式寻找Lambda2
        :param lambda1: 变量1
        :return:
        """
        E1 = self.E[i]
        # 判断当前所有Ei是否都相同 相同->随机选择 不相同->启发式选择
        if len(set(self.E)) == 1:
            # 所有Ei都相同 -> 随机选择
            index = i
            while index == i:
                index = np.random.randint(0, self.simpleSize)
            lambda2 = self.lambdas[index]
            E2 = self.E[index]
        else:
            if i == 0:
                index = 1
            else:
                index = 0
            # 所有Ei都不相同 -> 启发式选择
            if E1 > 0:
                # 如果E1>0, 选取最小的Ei为E2
                for j in range(self.simpleSize):
                    if self.E[j] < self.E[index] and index != i:
                        index = j
                lambda2 = self.lambdas[index]
                E2 = self.E[index]
            elif E1 < 0:
                # 如果E1<0, 选取最大的Ei为E2
                for j in range(self.simpleSize):
                    if self.E[j] > self.E[index] and index != i:
                        index = j
                lambda2 = self.lambdas[index]
                E2 = self.E[index]
            else:
                # 如果E1=0, 选取绝对值最大的Ei为E2
                for j in range(self.simpleSize):
                    if math.fabs(self.E[j]) > math.fabs(self.E[index]) and index != i:
                        index = j
                lambda2 = self.lambdas[index]
                E2 = self.E[index]

        return index, lambda2, E2

    def train(self, iter=100):
        """
        开始训练
        :param iter: 轮数
        """
        # iterStep：当前迭代次数，超过设置次数还未收敛则强制停止
        iterStep = 0
        # paramChanged：单次迭代中有参数改变则增加1
        paramChanged = 1
        while iterStep < iter and paramChanged > 0:
            # 打印当前迭代轮数
            print('iter:%d:%d' % (iterStep, iter))
            # 迭代步数加1
            iterStep += 1
            # 新的一轮将参数改变标志位重新置0
            paramChanged = 0
            for i in range(self.simpleSize):
                # 对于lambda1的选择，由于在大量训练集下去寻找“违反KKT条件最严重的支持向量或者样本”很费时，于是直接采用“首个违反KKT条件的样本的lambda”作为lambda1
                if self.isSatisfyKKT(i) is False:
                    index1 = i
                    lambda1Old = self.lambdas[index1]
                    E1Old = self.E[i]
                    # 寻找lambda2
                    index2, lambda2Old, E2Old = self.searchLambda2(i)
                    # 计算原始lambda2New
                    K11 = self.K[index1][index1]
                    K12 = self.K[index1][index2]
                    K22 = self.K[index2][index2]
                    KXi = K11 - 2 * K12 + K22
                    lambda2New = lambda2Old + self.Y[index2] * (E1Old - E2Old) / KXi
                    # 对lambda2New进行剪辑
                    if self.Y[index1] * self.Y[index2] == 1:
                        L = max(0, lambda1Old + lambda2Old - self.C)
                        H = min(self.C, lambda1Old + lambda2Old)
                    else:
                        L = max(-1 * lambda1Old + lambda2Old)
                        H = min(-1 * lambda1Old + lambda2Old + self.C, self.C)
                    if L == H:
                        # 如果两者相等，说明该变量无法再优化，直接跳到下一次循环
                        continue
                    if lambda2New > H:
                        lambda2New = H
                    elif lambda2New < L:
                        lambda2New = L

                    # 计算lambda1New
                    lambda1New = lambda1Old + self.Y[index1] * self.Y[index2] * (lambda2Old - lambda2New)
                    # 计算B
                    B1New = -1 * E1Old - self.Y[index1] * K11 * (lambda1New - lambda1Old) \
                            - self.Y[index2] * K12 * (lambda2New - lambda2Old) + self.B
                    B2New = -1 * E2Old - self.Y[index1] * K12 * (lambda1New - lambda1Old) \
                            - self.Y[index2] * K22 * (lambda2New - lambda2Old) + self.B
                    # -------------更新变量----------------#
                    # 更新B
                    if -1 * self.toler <= lambda1New <= self.C + self.toler:
                        self.B = B1New
                    elif -1 * self.toler <= lambda2New <= self.C + self.toler:
                        self.B = B2New
                    else:
                        self.B = (B1New + B2New) / 2
                    # 更新lambda
                    self.lambdas[index1] = lambda1New
                    self.lambdas[index2] = lambda2New

                    # 更新E（注意，更新E一定要在更新lambda1、lambda2之后）
                    self.E[index1] = self.calcEiNew(index1)
                    self.E[index2] = self.calcEiNew(index2)

                    # 如果α2的改变量过于小，就认为该参数未改变，不增加parameterChanged值
                    # 反之则自增1
                    if math.fabs(lambda2New - lambda2Old) >= 0.00001:
                        paramChanged += 1

                    # 打印迭代轮数，i值，该迭代轮数修改α数目
                    print("iter: %d i:%d, pairs changed %d" % (iterStep, i, paramChanged))

            # ---------------- 查找支持向量 ---------------------#
            # 清空supportVecIndex[]
            self.supportVecIndex.clear()
            for i in range(self.simpleSize):
                # 如果 0 < lambda < C，说明是支持向量
                if -1 * self.toler <= self.lambdas[i] <= self.C + self.toler:
                    # 将支持向量的索引保存起来
                    self.supportVecIndex.append(i)


def getMoonData(N, d=-2, r=10, w=2):
    N1 = 10 * N
    w2 = w / 2
    done = True
    data = np.empty(0)
    while done:
        # generate Rectangular data
        tmp_x = 2 * (r + w2) * (np.random.random([N1, 1]) - 0.5)
        tmp_y = (r + w2) * np.random.random([N1, 1])
        tmp = np.concatenate((tmp_x, tmp_y), axis=1)
        tmp_ds = np.sqrt(tmp_x * tmp_x + tmp_y * tmp_y)
        # generate double moon data ---upper
        idx = np.logical_and(tmp_ds > (r - w2), tmp_ds < (r + w2))
        idx = (idx.nonzero())[0]

        if data.shape[0] == 0:
            data = tmp.take(idx, axis=0)
        else:
            data = np.concatenate((data, tmp.take(idx, axis=0)), axis=0)
        if data.shape[0] >= N:
            done = False
    # print(data)
    db_moon = data[0:N, :]
    # print(db_moon)
    # generate double moon data ----down
    data_t = np.empty([N, 2])
    data_t[:, 0] = data[0:N, 0] + r
    data_t[:, 1] = -data[0:N, 1] - d
    db_moon = np.concatenate((db_moon, data_t), axis=0)
    Y = [1 if _ <= N else -1 for _ in range(2 * N)]
    Y = np.array(Y)
    # 打乱训练集
    index = np.random.permutation(2 * N)
    X = db_moon[index]
    Y = Y[index]
    return X, Y


def main():
    N = 100
    X, Y = getMoonData(N)


if __name__ == '__main__':
    main()
