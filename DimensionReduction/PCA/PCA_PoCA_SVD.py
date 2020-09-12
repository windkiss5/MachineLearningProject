# Python
# -*- coding: UTF-8 -*-
# @Time: 2020/9/12 20:13
# @Author: WINDKISS
# @File: PCA_PoCA_SVD.py
# @Function: What PCA_PoCA_SVD.py can do?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DimensionReduction.PCA import PCA

class PCA_PoCA_SVD:
    def __init__(self, X):
        self.X = X

    # 奇异值分解
    def PCA_SVD(self, discardDimensions=1):
        # 计算均值
        mu = np.mean(self.X, axis=0)
        # 中心化
        self.X = self.X - mu
        # 奇异值分解
        U, sv, V = np.linalg.svd(self.X, full_matrices=False)
        # 右特征矩阵-也即方差的特征矩阵
        V = V.T
        print("排序前的右特征矩阵V")
        print(pd.DataFrame(V))
        # 特征值计算
        sv = np.diag(sv)
        Singular = np.zeros(self.X.shape)
        Singular[0:sv.shape[0],0:sv.shape[1]] = sv
        # 特征值矩阵
        mu = Singular.T.dot(Singular)
        # 提取特征值
        mu = np.diagonal(mu)
        print("svd计算的特征值")
        print(mu)
        # 升序排序
        index = np.argsort(mu)
        V = V[:, index]
        print("排序后的右特征矩阵V")
        print(pd.DataFrame(V))
        # 重构
        resX = self.X.dot(V)
        print(V.shape)
        plt.figure('重构数据')
        ax = plt.gca(projection='3d')
        plt.title('重构数据', fontsize=14, fontproperties="SimHei")
        plt.tick_params(labelsize=10)
        ax.scatter(resX[:, 0], resX[:, 1], resX[:, 2], s=60, cmap='jet_r',
                   alpha=0.5, marker='o')
        plt.show()

        # 测试
        # 提取X.T.dot(X)的特征值,特征向量
        eigvals, eigvecs = np.linalg.eig(self.X.T.dot(self.X))
        print("X.T.dot(X)的特征值")
        print(pd.DataFrame(eigvals))
        print("X.T.dot(X)的特征矩阵")
        print(pd.DataFrame(eigvecs))
        # 降维
        return resX[:, discardDimensions:]

def main():
    # # 按高斯分布随机生成数据(以三维数据为例)
    # X = np.random.normal(loc=100, scale=1, size=(100, 3))
    # plt.figure('原始数据')
    # ax = plt.gca(projection='3d')
    # plt.title('原始数据', fontsize=14, fontproperties = "SimHei")
    # plt.tick_params(labelsize=10)
    # ax.scatter(X[:,0], X[:,1], X[:,2], s=60, cmap='jet_r',
    #            alpha=0.5, marker='o')
    # plt.show()
    # # PCA_SVD
    # resX = PCA_PoCA_SVD(X).PCA_SVD()
    # plt.figure('降维数据')
    # plt.title('降维数据', fontsize=14, fontproperties = "SimHei")
    # plt.plot(resX[:, 0], resX[:, 1], "*")
    # plt.show()
    # # 对比PCA_EVD
    # re = PCA.PCA(X).PCA_EVD()
    # plt.figure('降维数据2')
    # plt.title('降维数据2', fontsize=14, fontproperties="SimHei")
    # plt.plot(re[:, 0], re[:, 1], "*")
    # plt.show()
    A = np.array([[4, 2, -5], [6, 4, -9], [5, 3, -7]])
    ev, ei = np.linalg.eig(A)
    print("特征值")
    print(pd.DataFrame(ev))
    print("特征矩阵")
    print(pd.DataFrame(ei))
    print("============================================")
    X = np.random.normal(loc=100, scale=1, size=(100, 3))
    # X =np.array([[1, 0, 0, 0], [0, 0, 0, 4], [0, 3, 0, 0], [0, 0, 0 ,0], [2, 0, 0, 0]])
    # print(pd.DataFrame(X))
    # # 计算均值
    mu = np.mean(X, axis=0)
    # 中心化
    X = X - mu
    print("中心化")
    print(pd.DataFrame(X))



    # 特征值分解
    # 计算方差矩阵
    sigma = np.cov(X, rowvar=False)
    # 提取sigma的特征值,特征向量
    ev, ei = np.linalg.eig(sigma)
    print("特征值")
    print(pd.DataFrame(ev))

    print("特征矩阵")
    print(pd.DataFrame(ei))
    # # 奇异值分解(直接调用)
    # U, sv, V = np.linalg.svd(X, full_matrices=False)
    # print("(直接调用)奇异值")
    # print(pd.DataFrame(sv))
    # print("(直接调用)右奇异矩阵")
    # print(pd.DataFrame(V.T))
    # 奇异值分解(公式)
    eigvals, eigvecs = np.linalg.eig(X.T.dot(X))
    eigvals = np.sqrt(eigvals)
    print("(公式)奇异值")
    print(pd.DataFrame(eigvals))
    print("(公式)右奇异矩阵")
    print(pd.DataFrame(eigvecs))

    sv = np.diag(eigvals)
    Singular = np.zeros(X.shape)
    Singular[0:sv.shape[0], 0:sv.shape[1]] = sv
    # 特征值矩阵
    mu = Singular.T.dot(Singular) / 100
    # 提取特征值
    mu = np.diagonal(mu)
    print("svd计算的特征值")
    print(pd.DataFrame(mu))


if __name__ == '__main__':
    main()