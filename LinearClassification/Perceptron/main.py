# -*- coding:utf-8 -*-
# @Time : 2020/8/29 15:59
# @Author : WINDKISS
# @File : main.py
# @Function : 感知机main()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from LinearClassification.Perceptron.PerceptronSolver import PerceptronSolver

# ---------------------------- #
# 按高斯分布生成数据
# ---------------------------- #
num = 100
# 类别 1 -> label=1
class1 = np.random.normal(loc=2.3, scale=0.7, size=(num, 2))
label1 = -1 * np.ones((num, 1))
classLabel1 = np.concatenate((class1, label1), axis=1)
# 类别 2 -> label=-1
class2 = np.random.normal(loc=-0.9, scale=0.5, size=(num, 2))
label2 =  np.ones((num, 1))
classLabel2 = np.concatenate((class2, label2), axis=1)
# 训练数据
# X = np.concatenate((class1, class2), axis=1)
X = np.vstack((class1, class2))
print(pd.DataFrame(X))
# 标签

Y = np.vstack((label1, label2))
print(Y)
print("========================================-----------------")
plt.figure()
plt.plot(class1[:, 0], class1[:, 1], "*")
plt.plot(class2[:, 0], class2[:, 1], "*")
Y7 = np.concatenate((X, Y), axis=1)
print(pd.DataFrame(Y7))

a = np.array([[1, 2, 3], [5, 6, 7]])
p = a[:, 2]
print(pd.DataFrame(p))

i = np.zeros(3)
print(i)
W = PerceptronSolver(X[:, 0], X[:, 1], Y).perceptronClassifyNormal()
print(W)
# 打印直线(这里数据是二元的)
x = np.linspace(-1, 3, 50)
y = W[1] * x + W[0]
plt.plot(x, y)
plt.show()
