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
        self.sampleSize = self.X.shape[0]
        self.features = self.X.shape[1]
    def PoCA(self, discardDimensions=1):
        # 计算均值
        mu = np.mean(self.X, axis=0)
        # 中心化
        self.X = self.X - mu
        plt.figure('原始数据')
        ax = plt.gca(projection='3d')
        plt.title('原始数据', fontsize=14, fontproperties="SimHei")
        plt.tick_params(labelsize=10)
        ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], s=60, cmap='jet_r',
                   alpha=0.5, marker='o')
        plt.show()
        # 奇异值分解(计算左奇异矩阵)
        # 其中D实际为HX的奇异值对角矩阵D`.dot(D`.T)
        # 所以D实际上就是T矩阵的特征值的N倍(N为样本数)
        D, U = np.linalg.eig(self.X.dot(self.X.T))
        print("=========")
        print(pd.DataFrame(D))
        print("=========")
        # 由于numpy计算特征值与特征向量用到了复数运算，需要取实部操作
        D = np.real(D)
        D = np.sqrt(D)

        # 升序排序
        index = np.argsort(D)
        U = U[:, index]
        D = D[index]
        # 获取D`
        U = np.real(U)
        D = D[0: min(self.features, self.sampleSize)] / self.sampleSize
        D2 = np.diag(D)
        D = np.zeros(self.X.shape)
        D[D.shape[0] - D2.shape[0]: D.shape[0],
            D.shape[1] - D2.shape[1]: D.shape[1]] = D2
        # 获取重构后的坐标
        resX = U.dot(D)

        plt.figure('重构数据')
        ax = plt.gca(projection='3d')
        plt.title('重构数据', fontsize=14, fontproperties="SimHei")
        plt.tick_params(labelsize=10)
        ax.scatter(resX[:, 0], resX[:, 1], resX[:, 2], s=60, cmap='jet_r',
                   alpha=0.5, marker='o')
        plt.show()
        # 主坐标
        return resX[:, discardDimensions:]


def main():
    # 按高斯分布随机生成数据(以三维数据为例)
    # X = np.random.normal(loc=100, scale=1, size=(100, 3))
    X = np.array([[-1,-1,0,2,1],[2,0,0,-1,-1],[2,0,1,1,0]])
    plt.figure('原始数据')
    ax = plt.gca(projection='3d')
    plt.title('原始数据', fontsize=14, fontproperties = "SimHei")
    plt.tick_params(labelsize=10)
    ax.scatter(X[:,0], X[:,1], X[:,2], s=60, cmap='jet_r',
               alpha=0.5, marker='o')
    plt.show()
    # PCoA_SVD
    resX = PCoA_SVD(X).PoCA(3)
    print("PCoA_SVD")
    print(pd.DataFrame(resX))
    plt.figure('降维数据')
    plt.title('降维数据', fontsize=14, fontproperties = "SimHei")
    plt.plot(resX[:, 1], resX[:, 0], "*")
    plt.show()
    # 对比PCA_EVD
    re = PCA.PCA(X).PCA_EVD(3)
    print("PCA_EVD")
    print(pd.DataFrame(re))
    plt.figure('降维数据2')
    plt.title('降维数据2', fontsize=14, fontproperties="SimHei")
    plt.plot(re[:, 1], re[:, 0], "*")
    plt.show()

if __name__ == '__main__':
    main()
