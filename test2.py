import matplotlib.pyplot as plt
import numpy as np
from array import array


def mcmc(Pi, Q, N=1000, Nlmax=100000, isMH=False):
    X0 = np.random.randint(len(Pi))  # 第一步：从均匀分布（随便什么分布都可以）采样得到初始状态值x0
    T = N + Nlmax - 1
    result = [0 for i in range(T)]
    t = 0
    while t < T - 1:
        t = t + 1
        # 从条件概率分布Q(x|xt)中采样得到样本x∗
        # 该步骤是模拟采样，根据多项分布，模拟走到了下一个状态
        # （也可以将该步转换成一个按多项分布比例的均匀分布来采样）
        x_cur = np.argmax(np.random.multinomial(1, Q[result[t - 1]]))  # 第二步：取下一个状态 ，采样候选样本
        if isMH:
            '''
                细致平稳条件公式：πi Pij=πj Pji,∀i,j
            '''
            a = (Pi[x_cur] * Q[x_cur][result[t - 1]]) / (Pi[result[t - 1]] * Q[result[t - 1]][x_cur])  # 第三步：计算接受率
            acc = min(a, 1)
        else:  # mcmc
            acc = Pi[x_cur] * Q[x_cur][result[t - 1]]
        u = np.random.uniform(0, 1)  # 第四步：生成阈值
        if u < acc:  # 第五步：是否接受样本
            result[t] = x_cur
        else:
            result[t] = result[t - 1]
    return result


def count(q, n):
    L = array("d")
    l1 = array("d")
    l2 = array("d")
    for e in q:
        L.append(e)
    for e in range(n):
        l1.append(L.count(e))
    for e in l1:
        l2.append(e / sum(l1))
    return l1, l2


if __name__ == '__main__':
    Pi = np.array([0.5, 0.2, 0.3])  # 目标的概率分布
    # 状态转移矩阵，但是不满足在 平衡状态时和 Pi相符
    # 我们的目标是按照某种条件改造Q ，使其在平衡状态时和Pi相符
    # 改造方法就是，构造矩阵 P，且 P(i,j)=Q(i,j)α(i,j)
    #                          α(i, j) = π(j)Q(j, i)
    #                          α(j, i) = π(i)Q(i, j)
    Q = np.array([[0.9, 0.075, 0.025],
                  [0.15, 0.8, 0.05],
                  [0.25, 0.25, 0.5]])

    a = mcmc(Pi, Q)
    l1 = ['state%d' % x for x in range(len(Pi))]
    plt.pie(count(a, len(Pi))[0], labels=l1, labeldistance=0.3, autopct='%1.2f%%')
    plt.title("markov:" + str(Pi))
    plt.show()
