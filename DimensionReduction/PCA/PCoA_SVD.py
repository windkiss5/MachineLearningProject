# Python
# -*- coding: UTF-8 -*-
# @Time: 2020/9/13 11:19
# @Author: WINDKISS
# @File: PCoA_SVD.py
# @Function: 主坐标分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DimensionReduction.PCA import PCA

class PCoA_SVD:
    def __init__(self, X):
        self.X = X

    def PoCA(self, discardDimensions=1):
        # 计算均值
        mu = np.mean(self.X, axis=0)
        # 中心化
        self.X = self.X - mu
        # 奇异值分解(计算左奇异矩阵)
        # 其中D实际为HX的奇异值对角矩阵D`.dot(D`.T)
        # 所以D实际上就是T矩阵的特征值的N倍(N为样本数)
        D, U = np.linalg.eig(self.X.dot(self.X.T))
        # 由于numpy计算特征值与特征向量用到了复数运算，需要取实部操作
        D = np.real(D)
        U = np.real(U)
        # 升序排序
        index = np.argsort(D)
        D = D[index]
        U = U[:, index]
        plt.figure('重构数据')
        ax = plt.gca(projection='3d')
        plt.title('重构数据', fontsize=14, fontproperties="SimHei")
        plt.tick_params(labelsize=10)
        ax.scatter(U[:, 0], U[:, 1], U[:, 2], s=60, cmap='jet_r',
                   alpha=0.5, marker='o')
        plt.show()
        # 主坐标
        resX = U
        return resX[:, discardDimensions:]

def main():
    # 按高斯分布随机生成数据(以三维数据为例)
    X = np.random.normal(loc=100, scale=1, size=(100, 3))
    plt.figure('原始数据')
    ax = plt.gca(projection='3d')
    plt.title('原始数据', fontsize=14, fontproperties = "SimHei")
    plt.tick_params(labelsize=10)
    ax.scatter(X[:,0], X[:,1], X[:,2], s=60, cmap='jet_r',
               alpha=0.5, marker='o')
    plt.show()
    # PCoA_SVD
    resX = PCoA_SVD(X).PoCA()
    plt.figure('降维数据')
    plt.title('降维数据', fontsize=14, fontproperties = "SimHei")
    plt.plot(resX[:, 0], resX[:, 1], "*")
    plt.show()
    # 对比PCA_EVD
    re = PCA.PCA(X).PCA_EVD()
    plt.figure('降维数据2')
    plt.title('降维数据2', fontsize=14, fontproperties="SimHei")
    plt.plot(re[:, 0], re[:, 1], "*")
    plt.show()

if __name__ == '__main__':
    main()
