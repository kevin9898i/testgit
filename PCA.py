# -*- coding:utf-8 -*-
"""pca for fault detection and fault diagnosis"""
import os
import sys

import numpy as np
from pylab import *
from scipy.linalg import eigh
from scipy.stats import f, chi2, zscore
import seaborn as sns
import matplotlib.pyplot as plt
import load_data

config = {
"font.family":'serif',
"font.size": 20,
"mathtext.fontset":'stix',
"font.serif": ['SimSun'],
}
rcParams.update(config)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class PCA_FaultDetection():
    def __init__(self, cumper=0.95, signifi=0.99):
        self.signifi = signifi
        self.cumper = cumper
        self.model = None

    def normalize(self, X, Y):
        # 用训练数据的标准差和均值对测试数据标准化
        X_mu = np.mean(X, axis=0).reshape((1, X.shape[1]))
        X_std = np.std(X, axis=0).reshape((1, X.shape[1]))
        Xs = (X - X_mu) / X_std
        mu_array = np.ones((Y.shape[0], 1)) * X_mu
        st_array = np.ones((Y.shape[0], 1)) * X_std
        Ys = (Y - mu_array) / st_array
        return Xs, Ys

    def pc_number(self, X):
        # 确定主成分个数
        U, S, V = np.linalg.svd(X)
        print(S)
        if S.shape[0] == 1:
            i = 1
        else:
            i = 0
            var = 0
            while var < self.cumper * sum(S * S):
                var += S[i] * S[i]
                i += 1
            return i

    def train(self, X):
        row = X.shape[0]

        # 主成分数选择
        pc_num = self.pc_number(X)
        cov = X.T @ X / (len(X) - 1)
        U, S, V = np.linalg.svd(cov)
        P = U[:, :pc_num]
        T = X @ P
        lambdas = np.diag(S[:pc_num])

        # 计算控制限
        T2_limit = pc_num * (row - 1) / (row - pc_num) \
                   * f.ppf(self.signifi, pc_num, row - pc_num)

        SPE = np.square(X).sum(axis=1) - np.square(T).sum(axis=1)
        m, s = np.mean(SPE), np.var(SPE)
        g, h = s / (2 * m), 2 * np.square(m) / s
        SPE_limit = g * chi2.ppf(self.signifi, h)

        # 存储训练模型
        self.model = {
            'P': P,  # 负载矩阵
            'lambdas': lambdas,  # 特征值
            'num_pc': pc_num,   # 主成分个数
            'row': row,
            'SPE_limit': SPE_limit,
            'T2_limit': T2_limit

        }

    def isolation_pre(self, X):
        # 计算y, z
        pc_num = self.model['num_pc']
        lambdas = self.model['lambdas']
        P = self.model['P']
        A = lambdas[:pc_num, :pc_num]
        L = linalg.cholesky(A)
        y = (P @ L).T @ X.T
        Z = (P @ L).T @ P

        # 存储数据
        self.model['y'] = y
        self.model['Z'] = Z

        return y, Z

    def test(self, testdata):
        # 计算T2\SPE统计量
        lambdas = self.model['lambdas']
        P = self.model['P']
        H = np.identity(testdata.shape[1]) - P @ P.T
        T2 = []
        SPE = []
        for i in range(testdata.shape[0]):
            t1 = testdata[i, :] @ P
            t2 = testdata[i, :] @ H
            T2.append(t1 @ np.linalg.inv(self.model['lambdas']) @ t1.T)
            SPE.append(t2 @ t2.T)

        # 报警值个数
        SPE_alarm = np.nonzero(SPE > self.model['SPE_limit'])
        T2_alarm = np.nonzero(T2 > self.model['T2_limit'])
        print('测试样本总数：%d\n' % testdata.shape[0])
        print('SPE统计量报警总数：%d\n' % len(SPE_alarm[0]))
        print('T2统计量报警总数：%d\n' % len(T2_alarm[0]))

        result = {
            'SPE': SPE,
            'T2': T2
        }

        return result

    # 贡献图
    def single_sample_con(self, x_test):
        m = x_test.shape[1]
        for i in range(3):
            if i == 0:
                M1 = (self.model['P'] @ (np.linalg.inv(self.model['lambdas']) ** 0.5) @ self.model['P'].T)
            else:  # Qx
                M1 = np.identity(m) - self.model['P'] @ self.model['P'].T
            con = []
            for j in range(m):
                con += list(np.power(M1[j, :] @ (x_test.T), 2));
            if i == 0:  # Tx
                Tx_con = con;
            else:  # Qx
                Qx_con = con;

        singlesample_con_result = {
            'Tx_con': Tx_con,
            'Qx_con': Qx_con
        }
        return singlesample_con_result

    def con_bar_vis(self, con_result, fea_names):
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        Tx_con, Qx_con = con_result['Tx_con'], con_result['Qx_con']

        plt.figure(figsize=(10, 6), dpi=600)
        ax1 = plt.subplot(2, 1, 1)
        ax1.tick_params(axis='x', labelsize=6)  # 设置x轴标签大小
        ax1.bar(x=range(len(Tx_con)), height=Tx_con, width=0.7, label='Tx_con变量贡献')
        ax1.legend(loc="right")

        ax2 = plt.subplot(2, 1, 2)
        ax2.tick_params(axis='x', labelsize=10, rotation=-15)  # 设置x轴标签大小
        ax2.bar(x=fea_names, height=Qx_con, width=0.9, label='Qx_con变量贡献')
        ax2.legend(loc="right")

    def con_bar_vis(self, con_result):
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        Tx_con, Qx_con = con_result['Tx_con'], con_result['Qx_con']

        plt.figure(figsize=(9.6, 6.4), dpi=600)
        ax1 = plt.subplot(2, 1, 1)
        ax1.tick_params(axis='x', labelsize=6)  # 设置x轴标签大小
        ax1.bar(x=range(len(Tx_con)), height=Tx_con, width=0.7, label='Tx_con变量贡献')
        ax1.legend(loc="right")

        ax2 = plt.subplot(2, 1, 2)
        ax2.tick_params(axis='x', labelsize=10, rotation=-15)  # 设置x轴标签大小
        ax2.bar(x=range(len(Tx_con)), height=Qx_con, width=0.9, label='Qx_con变量贡献')
        ax2.legend(loc="right")

    def single_sample_recon(self, x_test):  # 贡献图(reconstruction based contribution plot)
        m = x_test.shape[1]
        for i in range(2):
            x_test = x_test.reshape(1, -1)
            if i == 0:  # Tc
                M1 = self.model['P'] @ np.linalg.inv(self.model['lambdas']) @ self.model['P'].T;
            else:  # Qx
                M1 = np.identity(m) - self.model['P'] @ self.model['P'].T
            Recon = []
            for j in range(m):
                Recon += list(np.power(M1[j, :] @ (x_test.T), 2) / M1[j, j])
            if i == 0:  # Tc
                Tx_recon = Recon;
            else:  # Qx
                Qx_recon = Recon;

        singlesample_recon_result = {
            'Tx_recon': Tx_recon,
            'Qx_recon': Qx_recon
        }
        return singlesample_recon_result

    def multi_sample_recon(self, X_test):  # 贡献图(reconstruction based contribution plot)
        n = X_test.shape[0]
        Tx_recon = []
        Qx_recon = []
        for i in range(n):
            singlesample_recon_result = self.single_sample_recon(X_test[i:i + 1, :])
            Tx_recon.append(singlesample_recon_result['Tx_recon'])
            Qx_recon.append(singlesample_recon_result['Qx_recon'])

        multisample_recon_result = {
            'Tx_recon': Tx_recon,
            'Qx_recon': Qx_recon
        }
        return multisample_recon_result

    def recon_vis_headmap(self, recon_result):
        plt.figure(figsize=(9.6, 6.4), dpi=200)
        ax1 = sns.heatmap(np.array(recon_result['Tx_recon']).T, cmap=sns.color_palette("RdBu_r", 50))
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Variables')

    #         ax2 = sns.heatmap(np.array(recon_result['Qx_recon']).T, cmap=sns.color_palette("RdBu_r", 50))
    #         ax2.set_xlabel('Samples')
    #         ax2.set_ylabel('Variables')

    def visualization(self, model, testresult):
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        SPE, T2 = testresult['SPE'], testresult['T2']
        SPE_limit, T2_limit = self.model['SPE_limit'], self.model['T2_limit']
        plt.figure(figsize=(10, 7))
        ax1 = plt.subplot(2, 1, 1)
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlim(1, 1010)
        plt.xticks(np.arange(100, 1010, 100))
        plt.tick_params(labelsize=20)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        # ax1.axhline(y=T2_limit, ls="--", color="r", label='T2控制限')
        plt.plot(np.arange(1, 1+len(T2)), T2_limit * np.ones((2160, 1)), color="r")
        # ax1.axhline(y=T2_limit, color="orangered", label='99%limit', linestyle='--')
        ax1.plot(np.arange(1, 1+len(T2)), T2, color='k')
        plt.ylabel('T$^2$', fontsize=20, fontname='Times New Roman')
        # plt.legend()
        # ax1.legend(loc="right")

        ax2 = plt.subplot(2, 1, 2)
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        plt.xlim(1, 1010)
        plt.ylim(-5, 45)
        plt.xticks(np.arange(100, 1010, 100))
        ax2.plot(np.arange(1, 1+len(SPE)), SPE, color='k')
        # ax2.axhline(y=SPE_limit, ls="--", color="r", label='SPE控制限')
        # ax2.axhline(y=SPE_limit, color="orangered", label='99%limit', linestyle='--')
        plt.plot(np.arange(1, 1+len(SPE)), SPE_limit * np.ones((2160, 1)), color="r")
        # ax2.set_title('过程监控图')
        # ax2.legend(loc="right")
        plt.ylabel('SPE', fontsize=20, fontname='Times New Roman')
        plt.xlabel('Sample', fontsize=20, fontname='Times New Roman')
        plt.tick_params(labelsize=20)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        os.makedirs('../figure/')
        plt.savefig("../figure/PCA.png", dpi=1000, bbox_inches='tight')
        # plt.legend()
        plt.show()

    def recon_bar_vis(self, recon_result, fea_names):
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        Tx_recon, Qx_recon = recon_result['Tx_recon'], recon_result['Qx_recon']
        plt.figure(figsize=(9.6, 6.4), dpi=600)
        ax1 = plt.subplot(2, 1, 1)
        ax1.bar(x=range(len(Tx_recon)), height=Tx_recon, width=0.9, label='Tx_recon变量重构贡献')
        ax1.legend(loc="right")
        ax2 = plt.subplot(2, 1, 2)
        ax2.bar(x=fea_names, height=Qx_recon, width=0.9, label='Qx_recon变量重构贡献')
        ax2.tick_params(axis='x', labelsize=10, rotation=-15)  # 设置x轴标签大小
        ax2.legend(loc="right")

    def recon_bar_vis(self, recon_result):
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        Tx_recon, Qx_recon = recon_result['Tx_recon'], recon_result['Qx_recon']
        plt.figure(figsize=(9.6, 6.4), dpi=600)
        ax1 = plt.subplot(2, 1, 1)
        ax1.bar(x=range(len(Tx_recon)), height=Tx_recon, width=0.9, label='Tx_recon变量重构贡献')
        ax1.legend(loc="right")
        ax2 = plt.subplot(2, 1, 2)
        ax2.bar(x=range(len(Tx_recon)), height=Qx_recon, width=0.9, label='Qx_recon变量重构贡献')
        ax2.tick_params(axis='x', labelsize=10, rotation=-15)  # 设置x轴标签大小
        ax2.legend(loc="right")

    def multi_sample_con(self, X_test):  # 贡献图(reconstruction based contribution plot)
        n = X_test.shape[0]
        Tx_con = []
        Qx_con = []
        for i in range(n):
            #             print(i)
            singlesample_con_result = self.single_sample_con(X_test[i:i + 1, :])
            Tx_con.append(singlesample_con_result['Tx_con'])
            Qx_con.append(singlesample_con_result['Qx_con'])
        multisample_con_result = {
            'Tx_con': Tx_con,
            'Qx_con': Qx_con
        }
        return multisample_con_result

    def con_vis_headmap(self, con_result):
        plt.figure(figsize=(9.6, 6.4), dpi=200)
        ax = sns.heatmap(np.array(con_result['Tx_con']).T, cmap=sns.color_palette("RdBu_r", 50))
        #         sns.set()
        #         ax = sns.heatmap(np.array(con_result['Qx_con']).T, cmap=sns.color_palette("RdBu_r", 50))
        ax.set_xlabel('Samples')
        ax.set_ylabel('Variables')

if __name__ == '__main__':
    X_Train, X_test = load_data.load_data()
    # print(X_Train, '\n', X_test)
    X_Train = X_Train.drop(['datetime'], axis=1)
    X_test = X_test.drop(['datetime'], axis=1)
    X_Train = np.array(X_Train)
    X_test = np.array(X_test)
    model = PCA_FaultDetection(cumper=0.85, signifi=0.99)
    # 数据标准化（若是标准化过后的数据则无需这一步）
    [X_Train, X_test] = model.normalize(X_Train, X_test)
    # 训练模型
    model.train(X_Train)
    # 代入测试数据
    testresult = model.test(X_test)
    # 检测结果可视化
    model.visualization(model, testresult)




