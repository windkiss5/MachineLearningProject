# Python
# -*- coding: UTF-8 -*-
# @Time: 2020/9/12 18:21
# @Author: WINDKISS
# @File: PCA.py
# @Function: PCA主成分分析
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, X):
        self.X = X

    # 特征值分解
    def PCA_EVD(self, discardDimensions = 1):
        # 计算均值
        mu = np.mean(self.X, axis=0)
        # 中心化
        self.X = self.X - mu
        # 计算方差矩阵
        sigma = np.cov(self.X, rowvar=False)
        # 提取sigma的特征值,特征向量
        eigvals, eigvecs = np.linalg.eig(sigma)
        print("排序前的特征值")
        print(pd.DataFrame(eigvals))
        print("排序前的特征矩阵")
        print(pd.DataFrame(eigvecs))
        # 升序排序
        index =np.argsort(eigvals)
        eigvals = eigvals[index]
        eigvecs = eigvecs[:,index]
        print("排序后的特征向量")
        print(pd.DataFrame(eigvecs))
        # 重构
        resX = self.X.dot(eigvecs)
        plt.figure('重构数据')
        ax = plt.gca(projection='3d')
        plt.title('重构数据', fontsize=14, fontproperties = "SimHei")
        plt.tick_params(labelsize=10)
        ax.scatter(resX[:, 0], resX[:, 1], resX[:, 2], s=60, cmap='jet_r',
                   alpha=0.5, marker='o')
        plt.show()
        # 降维
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
    # PCA
    resX = PCA(X).PCA_EVD()
    plt.figure('降维数据')
    plt.title('降维数据', fontsize=14, fontproperties = "SimHei")
    plt.plot(resX[:,1 ], resX[:,0], "*")
    plt.show()



if __name__ == '__main__':
    main()

