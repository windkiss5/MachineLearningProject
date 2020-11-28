# Python
# -*- coding: UTF-8 -*-
# @Time: 2020/10/4 21:12
# @Author: WINDKISS
# @File: SVM.py
# @Function: What SVM.py can do?
"""
数据集：Mnist
训练集数量：60000(实际使用：1000)
测试集数量：10000（实际使用：100)
------------------------------
"""
import time
import numpy as np
import math
import pandas as pd


class SVM:
    def __init__(self, X, Y, C=200, toler=0.001, sigma=10):
        """
        SVM类 本SVM类仅实现了高斯核下对双月分类数据集的分类
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
        # sampleSize：训练集数量    features：样本特征数目
        self.sampleSize, self.features = np.shape(self.X)
        # lambdas: 待优化拉格朗日算子
        self.lambdas = [0.] * self.sampleSize
        # E: Ej = f(xj) - yj
        self.E = -1. * self.Y
        # 支持向量下标集
        self.supportVecIndex = []
        # 高斯核函数矩阵(RBF核)
        self.K = self.calcKernel()
        # 标记是否训练
        self.isRunning = False

    def calcKernel(self):
        """
        计算核函数矩阵
        :return: 高斯核函数矩阵
        """
        K = [[0. for _ in range(self.sampleSize)] for _ in range(self.sampleSize)]
        for i in range(self.sampleSize):
            # 取出下标为i的样本
            Xi = self.X[i, :]
            for j in range(i, self.sampleSize):
                Xj = self.X[j, :]
                # 先计算||Xi - Xj||^2
                result = (Xi - Xj).T.dot((Xi - Xj))
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
        result = self.calcFxj(j) - self.Y[j]
        return result

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
                index = np.random.randint(0, self.sampleSize)
            lambda2 = self.lambdas[index]
            E2 = self.E[index]
        else:
            # 所有Ei都不相同 -> 启发式选择
            # 选择|Ei - Ej|最大对应的那个j作为index
            maxE1_E2 = -1
            index = 0
            for j in range(self.sampleSize):
                tmpE1_E2 = math.fabs(E1 - self.E[j])
                if tmpE1_E2 > maxE1_E2:
                    maxE1_E2 = tmpE1_E2
                    index = j

            lambda2 = self.lambdas[index]
            E2 = self.E[index]

        return index, lambda2, E2

    def train(self, iter=100):
        """
        开始训练
        :param iter: 轮数
        """
        self.isRunning = True
        # iterStep：当前迭代次数，超过设置次数还未收敛则强制停止
        iterStep = 0
        # paramChanged：单次迭代中有参数改变则增加1
        paramChanged = 1
        while iterStep < iter and paramChanged > 0 and self.isRunning is True:
            # 打印当前迭代轮数
            print('iter:%d:%d' % (iterStep, iter))
            # 迭代步数加1
            iterStep += 1
            # 新的一轮将参数改变标志位重新置0
            paramChanged = 0
            for i in range(self.sampleSize):
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
                        L = max(0, -1 * lambda1Old + lambda2Old)
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
            for i in range(self.sampleSize):
                # 如果 0 < lambda < C，说明是支持向量
                if self.toler <= self.lambdas[i] <= self.C + self.toler:
                    # 将支持向量的索引保存起来
                    self.supportVecIndex.append(i)

    def calcSinglKernel(self, x, xi):
        x = np.array(x)
        xi = np.array(xi)
        result = (x - xi).dot((x - xi).T)
        result = np.exp(-1 * result / (2 * self.sigma ** 2))
        return result

    def predict(self, x):
        """
        预测单个样本的分类结果
        :param x: 样本
        :return: 分类结果 {-1， 1}
        """
        result = 0
        # 这里只考虑支持向量的原因是在于分类正确的样本对应的lambda恒为0
        for i in self.supportVecIndex:
            # 先单独将核函数计算出来
            tmp = self.calcSinglKernel(x, self.X[i, :])
            # 对每一项子式进行求和，最终计算得到求和项的值
            result += self.lambdas[i] * self.Y[i] * tmp
        result += self.B
        return np.sign(result)

    def test(self, X, Y):
        X = np.array(X)
        # 错误计数值
        errorCnt = 0
        # 遍历测试集所有样本
        for i in range(len(X)):
            # 获取预测结果
            result = self.predict(X[i, :].T)
            # 打印目前进度
            print('test:%d:%d' % (i, len(X)), 'predict:', result, ' label:', Y[i])
            # 如果预测与标签不一致，错误计数值加一
            if result != Y[i]:
                errorCnt += 1
        # 返回正确率
        return 1 - errorCnt / len(X)


def loadData(fileName):
    '''
    加载文件
    :param fileName:要加载的文件路径
    :return: 数据集和标签集
    '''
    # 存放数据及标记
    dataArr = []
    labelArr = []
    # 读取文件
    fr = open(fileName)
    # 遍历文件中的每一行
    for line in fr.readlines():
        # 获取当前行，并按“，”切割成字段放入列表中
        # strip：去掉每行字符串首尾指定的字符（默认空格或换行符）
        # split：按照指定的字符将字符串切割成每个字段，返回列表形式
        curLine = line.strip().split(',')
        # 将每行中除标记外的数据放入数据集中（curLine[0]为标记信息）
        # 在放入的同时将原先字符串形式的数据转换为0-1的浮点型
        dataArr.append([int(num) / 255 for num in curLine[1:]])
        # 将标记信息放入标记集中
        # 放入的同时将标记转换为整型
        # 数字0标记为1  其余标记为-1
        if int(curLine[0]) == 0:
            labelArr.append(1)
        else:
            labelArr.append(-1)
    # 返回数据集和标记
    return dataArr, labelArr


# def getMoonData(N, d=-2, r=10, w=2):
#     N1 = 10 * N
#     w2 = w / 2
#     done = True
#     data = np.empty(0)
#     while done:
#         # generate Rectangular data
#         tmp_x = 2 * (r + w2) * (np.random.random([N1, 1]) - 0.5)
#         tmp_y = (r + w2) * np.random.random([N1, 1])
#         tmp = np.concatenate((tmp_x, tmp_y), axis=1)
#         tmp_ds = np.sqrt(tmp_x * tmp_x + tmp_y * tmp_y)
#         # generate double moon data ---upper
#         idx = np.logical_and(tmp_ds > (r - w2), tmp_ds < (r + w2))
#         idx = (idx.nonzero())[0]
#
#         if data.shape[0] == 0:
#             data = tmp.take(idx, axis=0)
#         else:
#             data = np.concatenate((data, tmp.take(idx, axis=0)), axis=0)
#         if data.shape[0] >= N:
#             done = False
#     # print(data)
#     db_moon = data[0:N, :]
#     # print(db_moon)
#     # generate double moon data ----down
#     data_t = np.empty([N, 2])
#     data_t[:, 0] = data[0:N, 0] + r
#     data_t[:, 1] = -data[0:N, 1] - d
#     db_moon = np.concatenate((db_moon, data_t), axis=0)
#     Y = [1 if _ <= N else -1 for _ in range(2 * N)]
#     Y = np.array(Y)
#
#     return db_moon, Y


def main():
    start = time.time()

    # 获取训练集及标签
    print('start read transSet')
    trainDataList, trainLabelList = loadData('C:/Users/WINDKISS/Downloads/data/mnist_train.csv')

    # 获取测试集及标签
    print('start read testSet')
    testDataList, testLabelList = loadData('C:/Users/WINDKISS/Downloads/data/mnist_test.csv')

    # 初始化SVM类
    print('start init SVM')
    X = np.array(trainDataList[:1000])
    Y = np.array(trainLabelList[:1000])
    # 打乱训练集
    index = np.random.permutation(1000)
    X = X[index]
    Y = Y[index]
    svm = SVM(X, Y)

    # 开始训练
    print('start to train')
    svm.train()

    # 开始测试
    print('start to test')
    accuracy = svm.test(testDataList[:100], testLabelList[:100])
    print('the accuracy is:%d' % (accuracy * 100), '%')

    # 打印时间
    print('time span:', time.time() - start)


if __name__ == '__main__':
    main()
