import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import BQ.版本2.Modules as M
import sys
import random
import pandas as pd #用于数据输出
from sympy import symbols, Eq, solve
import concurrent.futures
import time as T
import subprocess
import json
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'Times New Roman'

'''动态策略'''
# Time = np.empty([0])
# Data = np.empty([0, 6])
# for pha in range(1, 7):
#     data = np.load(r"D:\pythonfile\GWO3\BQ\版本2\policy\data\ours_NN_enemy_agent_dic_mak_Adv_phase=" + str(pha) + "_KE=0.14_KR=0.008_KO=0.65.npy")
#     time = np.load(r"D:\pythonfile\GWO3\BQ\版本2\policy\data\ours_NN_enemy_agent_dic_mak_time_Adv_phase=" + str(pha) + "_KE=0.14_KR=0.008_KO=0.65.npy")
#     Time = np.hstack([Time, time])
#     if(data.shape[1] != 6):
#         data = np.hstack([data, np.zeros([data.shape[0], 6-data.shape[1]])])
#     Data = np.vstack([Data, data])
#
# policy = np.empty([0, 6])
# time_len = max(Time) / 6
# for i in range(6):
#     index = np.where([(Time >= time_len * i) & (Time < time_len * (i + 1))])[-1]
#     data = Data[index, :]
#     data_sum = np.sum(data, axis=0)
#     data_sum = data_sum / np.sum(data_sum)
#     policy = np.vstack([policy, data_sum])
#
# color = ['#E6A326',  '#6EDB57', '#37A4F6', '#DB5F57', "#7F7F7F", "#548235"]
# Time = [(i+0.5) * time_len for i in range(6)]
# fig, ax = plt.subplots(figsize=(7.5 * 7.69, 7.5))
# for i in range(6):
#     plt.plot(Time, policy[:, i], marker="*", markersize=55, color=color[i], linewidth=8.0)
# plt.xlim(0.0, time_len * 6)
# plt.ylim(-0.05, 0.9)
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.grid()
# plt.tight_layout()
# plt.show()


color = ['#E6A326',  '#6EDB57', '#37A4F6', '#DB5F57', "#7F7F7F", "#548235"]
linestyle = ['-', '-', '-', '--']
file = ["D:\pythonfile\GWO4\BQ\版本2\policy\data\policy1_enemy_Adv_phase=", "D:\pythonfile\GWO4\BQ\版本2\policy\data\policy2_enemy_Adv_phase=",
        "D:\pythonfile\GWO4\BQ\版本2\policy\data\ours_enemy_Adv_phase=", "D:\pythonfile\GWO4\BQ\版本2\policy\data\ours_NN_enemy_Adv_phase="]


'''平均值'''
# fig, axes = plt.subplots(figsize=(7.5 * 2.32, 7.5))
# for fil in range(len(file)):
#     data_1_4 = np.empty([0, 5])
#     for i in range(1, 4):
#         data = np.load(file[fil] + str(i) + "_KE=0.14_KR=0.008_KO=0.65.npy",
#             allow_pickle=True)
#         data_1_4 = np.vstack([data_1_4, data])
#
#     data_1_6 = np.empty([0, 7])
#     for i in range(4, 7):
#         data = np.load(file[fil] + str(i) + "_KE=0.14_KR=0.008_KO=0.65.npy",
#             allow_pickle=True)
#         data_1_6 = np.vstack([data_1_6, data])
#
#     data_1_4_ave = np.sum(data_1_4[:, 1:], axis=1) / 4
#     data_1_4_ave = np.vstack([data_1_4[:, 0], data_1_4_ave])
#     data_1_6_ave = np.sum(data_1_6[:, 1:], axis=1) / 6
#     data_1_6_ave = np.vstack([data_1_6[:, 0], data_1_6_ave])
#     data_ave = np.hstack([data_1_4_ave, data_1_6_ave])
#     plt.plot(data_ave[0, :], data_ave[1, :], color=color[fil], linewidth=8.0, linestyle=linestyle[fil])
# plt.xlim(0.0, 48.0)
# plt.ylim(0.08, 2.5)
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.grid()
# plt.tight_layout()
# plt.show()

'''最大值'''
# fig, axes = plt.subplots(figsize=(7.5 * 2.32, 7.5))
# for fil in range(len(file)):
#     data_1_4 = np.empty([0, 5])
#     for i in range(1, 4):
#         data = np.load(file[fil] + str(i) + "_KE=0.14_KR=0.008_KO=0.65.npy",
#             allow_pickle=True)
#         data_1_4 = np.vstack([data_1_4, data])
#
#     data_1_6 = np.empty([0, 7])
#     for i in range(4, 7):
#         data = np.load(file[fil] + str(i) + "_KE=0.14_KR=0.008_KO=0.65.npy",
#             allow_pickle=True)
#         data_1_6 = np.vstack([data_1_6, data])
#
#     data_1_4_max = np.max(data_1_4[:, 1:], axis=1)
#     data_1_4_max = np.vstack([data_1_4[:, 0], data_1_4_max])
#     data_1_6_max = np.max(data_1_6[:, 1:], axis=1)
#     data_1_6_max = np.vstack([data_1_6[:, 0], data_1_6_max])
#     data_max = np.hstack([data_1_4_max, data_1_6_max])
#     plt.plot(data_max[0, :], data_max[1, :], color=color[fil], linewidth=8.0, linestyle=linestyle[fil])
#
# plt.xlim(0.0, 48.0)
# plt.ylim(0.1, 2.5)
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.grid()
# plt.tight_layout()
# plt.show()

'''最小值'''
# fig, axes = plt.subplots(figsize=(7.5 * 2.32, 7.5))
# for fil in range(len(file)):
#     data_1_4 = np.empty([0, 5])
#     for i in range(1, 4):
#         data = np.load(file[fil] + str(i) + "_KE=0.14_KR=0.008_KO=0.65.npy",
#             allow_pickle=True)
#         data_1_4 = np.vstack([data_1_4, data])
#
#     data_1_6 = np.empty([0, 7])
#     for i in range(4, 7):
#         data = np.load(file[fil] + str(i) + "_KE=0.14_KR=0.008_KO=0.65.npy",
#             allow_pickle=True)
#         data_1_6 = np.vstack([data_1_6, data])
#
#     data_1_4_min = np.min(data_1_4[:, 1:], axis=1)
#     data_1_4_min = np.vstack([data_1_4[:, 0], data_1_4_min])
#     data_1_6_min = np.min(data_1_6[:, 1:], axis=1)
#     data_1_6_min = np.vstack([data_1_6[:, 0], data_1_6_min])
#     data_min = np.hstack([data_1_4_min, data_1_6_min])
#     plt.plot(data_min[0, :], data_min[1, :], color=color[fil], linewidth=8.0, linestyle=linestyle[fil])
#
# plt.xlim(0.0, 48.0)
# plt.ylim(0.08, 2.5)
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.grid()
# plt.tight_layout()
# plt.show()


# 定义孙网络模型
class Grandson_net(nn.Module):
    def __init__(self, order):
        super(Grandson_net, self).__init__()
        self.order = order
        self.fc = nn.Sequential(
            nn.Linear(self.order, 1, bias=False),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.fc(x)
# 定义恢复子网络模型
class Son_recovernet(nn.Module):
    def __init__(self, channels, order):
        super(Son_recovernet, self).__init__()
        self.channels = channels
        self.order = order
        self.fc = Grandson_net(self.order)

    def forward(self, x):  # x.shape=[64, channels, order]
        outputs = torch.zeros((x.shape[0], self.channels, 1))
        for i in range(self.channels):
            net = self.fc
            outputs[:, i, :] = net(x[:, i, :])
        return outputs
# 定义交换子网络模型
class Son_swapnet(nn.Module):
    def __init__(self, channels, order, mat, diff_indices):
        super(Son_swapnet, self).__init__()
        self.channels = channels
        self.order = order
        self.mat = mat
        self.dim = int(self.channels*(self.channels-1)/2)
        self.diff_indices = diff_indices
        self.fc2 = Grandson_net(self.order)
        self.fc1 = DiagonalLinear(self.channels)
        self.fc3 = DiagonalLinear(self.dim)

    def forward(self, x):
        outputs1 = self.fc1(x)
        diffs = torch.matmul(outputs1, self.diff_indices)
        sign = torch.sign(diffs).reshape(-1, self.dim, 1)
        outputs11 = torch.abs(diffs)

        # 计算幂运算
        outputs2 = torch.stack([outputs11 ** (i + 1) for i in range(self.order)], dim=2)

        # 应用fc2
        outputs3 = torch.zeros([x.shape[0], self.dim, 1])
        for i in range(self.dim):
            outputs3[:, i, :] = self.fc2(outputs2[:, i, :])

        outputs3 = torch.mul(outputs3, sign)
        outputs3 = self.fc3(outputs3.reshape(-1, self.dim)).reshape(-1, self.dim, 1)

        outputs4 = torch.zeros([x.shape[0], self.channels, 1])
        for i in range(self.channels):
            outputs4[:, i, :] -= torch.sum(outputs3[:, self.mat[i][0], :], dim=1)
            outputs4[:, i, :] += torch.sum(outputs3[:, self.mat[i][1], :], dim=1)

        return outputs4
class DiagonalLinear(nn.Module):
    def __init__(self, channels, bias=False):
        super(DiagonalLinear, self).__init__()
        self.diagonal = nn.Parameter(torch.zeros(channels)+1.0)  # 可训练的对角线元素
    def forward(self, x):
        # 创建一个对角矩阵
        diag_mat = torch.diag_embed(self.diagonal)
        # 使用矩阵乘法（实际上是逐元素乘法，因为x乘以对角矩阵）
        out = torch.matmul(x, diag_mat)
        return out
# 定义网络模型
class Net(nn.Module):
    def __init__(self, channels, order, mat, diff_indices):
        super(Net, self).__init__()
        self.channels = channels
        self.order = order
        self.mat = mat
        self.diff_indices = diff_indices

        self.Son_swapnet = Son_swapnet(self.channels, self.order, self.mat, self.diff_indices)
        self.Son_recovernet = Son_recovernet(self.channels, self.order)

    def forward(self, x1, x2):  # x.shape=[64, channels, 1]
        outputs = self.Son_swapnet(x1) + self.Son_recovernet(x2)
        return outputs

'''PNi预测'''
# time_NN = [[0.1, 10.1, 10.1, 10.2, 10.3, 10.4, 10.5, 11.1, 11.1, 11.2, 11.3, 11.4, 18.1, 18.1, 18.2, 18.3, 18.4, 18.5, 30.1, 30.1, 30.2, 30.3, 30.4, 30.5, 31.1, 31.1, 31.2, 31.3, 31.4, 31.5, 48.0],
#            [18.1, 18.1, 18.2, 18.3, 18.4, 18.5, 30.1, 30.1, 30.2, 30.3, 30.4, 30.5, 31.1, 31.1, 31.2, 31.3, 31.4, 31.5, 48.0]]
# PNi_NN = [[1.0000186, 1.0000186, 0.8109761, 0.80340934, 0.8024688, 0.80221707, 0.8013827, 0.8013827, 0.86594045, 0.85602313, 0.85155934, 0.8540178, 0.8540178, 0.97380304, 0.90090805, 0.88973284, 0.9861171, 0.99391705, 0.99391705, 1.2006423, 1.2003194, 1.2002373, 1.2002013, 1.2002016, 1.2002016, 1.4079989, 1.4312227, 1.2634512, 1.2236387, 1.219728, 1.219728],
#           [1.0000273, 1.0000273, 1.5699819, 1.5905883, 1.5933568, 1.5938824, 1.5962403, 1.5962403, 1.4104712, 1.4336601, 1.4450914, 1.437385, 1.437385, 1.0493364, 1.0801908, 1.0237786, 0.98049736, 0.9873471, 0.9873471, 1.1998907, 1.1997125, 1.1997235, 1.1999038, 1.1999032, 1.1999032, 1.1770339, 1.0324218, 1.2066799, 1.2678947, 1.2698773, 1.2698773],
#           [0.9999763, 0.9999763, 0.8105163, 0.8032716, 0.80232495, 0.802123, 0.80130935, 0.80130935, 0.86396134, 0.85489863, 0.8515818, 0.85417545, 0.85417545, 1.0851951, 1.085805, 1.0189158, 0.9833123, 0.9887787, 0.9887787, 1.1998731, 1.1997998, 1.1998847, 1.199872, 1.1998973, 1.1998973, 1.13065, 1.1837871, 1.1771445, 1.1698704, 1.1703787, 1.1703787],
#           [0.99997777, 0.99997777, 0.80852574, 0.8027306, 0.8018495, 0.80177736, 0.8010677, 0.8010677, 0.85962707, 0.8554178, 0.85176724, 0.85442156, 0.85442156, 0.9292953, 0.99045825, 1.0097455, 0.98090756, 0.9861068, 0.9861068, 0.80022526, 0.80053735, 0.80057406, 0.80051976, 0.8005129, 0.8005129, 0.7499177, 0.78469586, 0.7848259, 0.77996975, 0.78029025, 0.78029025],
#           [0.0, 0.95100564, 0.9242953, 1.0006546, 1.0218178, 1.0130315, 1.0130315, 0.79968435, 0.80015445, 0.8001646, 0.80010986, 0.80013204, 0.80013204, 0.7903999, 0.7828515, 0.7847558, 0.7798364, 0.78022337, 0.78022337],
#           [0.0, 1.0113643, 1.0183429, 1.0571728, 1.0473478, 1.0308188, 1.0308188, 0.7996844, 0.7994765, 0.7994158, 0.7993933, 0.79935277, 0.79935277, 0.7439998, 0.7850212, 0.7831426, 0.77878994, 0.7795023, 0.7795023]]
#
# time = [[0.0, 10.0, 10.0, 18.0, 18.0, 31.0, 31.0, 48.0], [18.0, 18.0, 31.0, 31.0, 48.0]]
# PNi = [[1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 1.2, 1.2],
#        [1.0, 1.0, 1.6, 1.6, 1.0, 1.0, 1.2, 1.2],
#        [1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 1.2, 1.2],
#        [1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 0.8, 0.8],
#        [0.0, 1.0, 1.0, 0.8, 0.8],
#        [0.0, 1.0, 1.0, 0.8, 0.8]]
#
# fig, axes = plt.subplots(2, 1, figsize=(7.5 * 2.32, 7.5), sharex=True, sharey=True)
# for i in range(0, 1):
#     for j in range(2):
#         ax = axes[j]
#         # ax = axes[i*2+j]
#         ax.plot(time[0], PNi[i*2+j], color="#37A4F6", linewidth=8.0, linestyle='-')
#         ax.plot(time_NN[0], PNi_NN[i*2+j], color="#DB5F57", linewidth=8.0, linestyle='--')
#         ax.plot([10, 10], [0.5, 0.55], color="black", linewidth=1.0, linestyle='-')
#         ax.plot([18, 18], [0.5, 0.55], color="black", linewidth=1.0, linestyle='-')
#         ax.plot([30, 30], [0.5, 0.55], color="black", linewidth=1.0, linestyle='-')
#         ax.plot([0, 0.4], [1.15, 1.15], color="black", linewidth=1.0, linestyle='-')
#
# for j in range(2):
#     ax = axes[j]
#     ax.set_xlim(0.0, 48.0)
#     ax.set_ylim(0.5, 1.8)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.tick_params(axis='x', labelsize=0.0)
#     ax.tick_params(axis='y', labelsize=0.0)
#
# plt.tight_layout()
# plt.show()
#
# fig, axes = plt.subplots(2, 1, figsize=(7.5 * 2.32, 7.5), sharex=True, sharey=True)
# for i in range(1, 2):
#     for j in range(2):
#         ax = axes[j]
#         # ax = axes[i*2+j]
#         ax.plot(time[0], PNi[i*2+j], color="#37A4F6", linewidth=8.0, linestyle='-')
#         ax.plot(time_NN[0], PNi_NN[i*2+j], color="#DB5F57", linewidth=8.0, linestyle='--')
#         ax.plot([10, 10], [0.5, 0.55], color="black", linewidth=1.0, linestyle='-')
#         ax.plot([18, 18], [0.5, 0.55], color="black", linewidth=1.0, linestyle='-')
#         ax.plot([30, 30], [0.5, 0.55], color="black", linewidth=1.0, linestyle='-')
#         ax.plot([0, 0.4], [1.15, 1.15], color="black", linewidth=1.0, linestyle='-')
#
# for j in range(2):
#     ax = axes[j]
#     ax.set_xlim(0.0, 48.0)
#     ax.set_ylim(0.5, 1.8)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.tick_params(axis='x', labelsize=0.0)
#     ax.tick_params(axis='y', labelsize=0.0)
# plt.tight_layout()
# plt.show()
#
#
# fig, axes = plt.subplots(2, 1, figsize=(7.5 * 2.32, 7.5), sharex=True, sharey=True)
# for i in range(2, 3):
#     for j in range(2):
#         ax = axes[j]
#         # ax = axes[i*2+j]
#         ax.plot(time[1], PNi[i*2+j], color="#37A4F6", linewidth=8.0, linestyle='-')
#         ax.plot(time_NN[1], PNi_NN[i*2+j], color="#DB5F57", linewidth=8.0, linestyle='--')
#         ax.plot([10, 10], [0.5, 0.55], color="black", linewidth=1.0, linestyle='-')
#         ax.plot([18, 18], [0.5, 0.55], color="black", linewidth=1.0, linestyle='-')
#         ax.plot([30, 30], [0.5, 0.55], color="black", linewidth=1.0, linestyle='-')
#         ax.plot([0, 0.4], [1.15, 1.15], color="black", linewidth=1.0, linestyle='-')
#
# for j in range(2):
#     ax = axes[j]
#     # ax = axes[i*2+j]
#     ax.set_xlim(0.0, 48.0)
#     ax.set_ylim(0.5, 1.8)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.tick_params(axis='x', labelsize=0.0)
#     ax.tick_params(axis='y', labelsize=0.0)
#     # ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
#
# plt.tight_layout()
# plt.show()







# plt.plot(time3, pni11, color="#37A4F6", linewidth=8.0, linestyle='-')
# plt.plot(time1, pni1, color="#DB5F57", linewidth=8.0, linestyle='--')
# plt.xlim(0.0, 45.0)
# plt.ylim(0.0, 2.0)
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# plt.xticks([])
# plt.yticks([])
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=2))
# plt.tight_layout()

# time1 = [0.1, 10.1, 10.1, 10.2, 10.3, 10.4, 10.5, 11.1, 11.1, 11.2, 11.3, 11.4, 25.1, 25.1, 25.2, 25.3, 25.4, 25.5, 35.1, 35.1, 35.2, 35.3, 35.4, 35.5, 36.1, 36.1, 36.2, 36.3, 36.4, 36.5, 45.0]
# pni1 = [1.0000186, 1.0000186, 0.8109761, 0.80340934, 0.8024688, 0.80221707, 0.8013827, 0.8013827, 0.86594045, 0.85602313, 0.85155934, 0.8540178, 0.8540178, 0.97380304, 0.90090805, 0.88973284, 0.9861171, 0.99391705, 0.99391705, 1.2006423, 1.2003194, 1.2002373, 1.2002013, 1.2002016, 1.2002016, 1.4079989, 1.4312227, 1.2634512, 1.2236387, 1.219728, 1.219728]
# pni2 = [1.0000273, 1.0000273, 1.5699819, 1.5905883, 1.5933568, 1.5938824, 1.5962403, 1.5962403, 1.4104712, 1.4336601, 1.4450914, 1.437385, 1.437385, 1.0493364, 1.0801908, 1.0237786, 0.98049736, 0.9873471, 0.9873471, 1.1998907, 1.1997125, 1.1997235, 1.1999038, 1.1999032, 1.1999032, 1.1770339, 1.0324218, 1.2066799, 1.2678947, 1.2698773, 1.2698773]
# pni3 = [0.9999763, 0.9999763, 0.8105163, 0.8032716, 0.80232495, 0.802123, 0.80130935, 0.80130935, 0.86396134, 0.85489863, 0.8515818, 0.85417545, 0.85417545, 1.0851951, 1.085805, 1.0189158, 0.9833123, 0.9887787, 0.9887787, 1.1998731, 1.1997998, 1.1998847, 1.199872, 1.1998973, 1.1998973, 1.13065, 1.1837871, 1.1771445, 1.1698704, 1.1703787, 1.1703787]
# pni4 = [0.99997777, 0.99997777, 0.80852574, 0.8027306, 0.8018495, 0.80177736, 0.8010677, 0.8010677, 0.85962707, 0.8554178, 0.85176724, 0.85442156, 0.85442156, 0.9292953, 0.99045825, 1.0097455, 0.98090756, 0.9861068, 0.9861068, 0.80022526, 0.80053735, 0.80057406, 0.80051976, 0.8005129, 0.8005129, 0.7499177, 0.78469586, 0.7848259, 0.77996975, 0.78029025, 0.78029025]
# time2 = [25.1, 25.1, 25.2, 25.3, 25.4, 25.5, 35.1, 35.1, 35.2, 35.3, 35.4, 35.5, 36.1, 36.1, 36.2, 36.3, 36.4, 36.5, 45.0]
# pni5 = [0.0, 0.95100564, 0.9242953, 1.0006546, 1.0218178, 1.0130315, 1.0130315, 0.79968435, 0.80015445, 0.8001646, 0.80010986, 0.80013204, 0.80013204, 0.7903999, 0.7828515, 0.7847558, 0.7798364, 0.78022337, 0.78022337]
# pni6 = [0.0, 1.0113643, 1.0183429, 1.0571728, 1.0473478, 1.0308188, 1.0308188, 0.7996844, 0.7994765, 0.7994158, 0.7993933, 0.79935277, 0.79935277, 0.7439998, 0.7850212, 0.7831426, 0.77878994, 0.7795023, 0.7795023]
#
#
# time3 = [0.0, 10.0, 10.0, 25.0, 25.0, 35.0, 35.0, 45.0]
# pni11 = [1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 1.2, 1.2]
# pni21 = [1.0, 1.0, 1.6, 1.6, 1.0, 1.0, 1.2, 1.2]
# pni31 = [1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 1.2, 1.2]
# pni41 = [1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 0.8, 0.8]
# time4 = [25.0, 25.0, 35.0, 35.0, 45.0]
# pni51 = [0.0, 1.0, 1.0, 0.8, 0.8]
# pni61 = [0.0, 1.0, 1.0, 0.8, 0.8]
#
# fig, ax = plt.subplots(figsize=(7.5 * 2.32, 7.5))
# plt.subplot(6, 1, 1)
# plt.plot(time3, pni11, color="#37A4F6", linewidth=8.0, linestyle='-')
# plt.plot(time1, pni1, color="#DB5F57", linewidth=8.0, linestyle='--')
# plt.xlim(0.0, 45.0)
# plt.ylim(0.0, 2.0)
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# plt.xticks([])
# plt.yticks([])
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=2))
# plt.tight_layout()
#
#
# plt.subplot(6, 1, 2)
# plt.plot(time3, pni21, color="#37A4F6", linewidth=8.0, linestyle='-')
# plt.plot(time1, pni2, color="#DB5F57", linewidth=8.0, linestyle='--')
# plt.xlim(0.0, 45.0)
# plt.ylim(0.0, 2.0)
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# plt.xticks([])
# plt.yticks([])
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=2))
# plt.tight_layout()
#
# plt.subplot(6, 1, 3)
# plt.plot(time3, pni31, color="#37A4F6", linewidth=8.0, linestyle='-')
# plt.plot(time1, pni3, color="#DB5F57", linewidth=8.0, linestyle='--')
# plt.xlim(0.0, 45.0)
# plt.ylim(0.0, 2.0)
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# plt.xticks([])
# plt.yticks([])
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=2))
# plt.tight_layout()
#
#
# plt.subplot(6, 1, 4)
# plt.plot(time3, pni41, color="#37A4F6", linewidth=8.0, linestyle='-')
# plt.plot(time1, pni4, color="#DB5F57", linewidth=8.0, linestyle='--')
# plt.xlim(0.0, 45.0)
# plt.ylim(0.0, 2.0)
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# plt.xticks([])
# plt.yticks([])
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=2))
# plt.tight_layout()
#
#
# plt.subplot(6, 1, 5)
# plt.plot(time4, pni51, color="#37A4F6", linewidth=8.0, linestyle='-')
# plt.plot(time2, pni5, color="#DB5F57", linewidth=8.0, linestyle='--')
# plt.xlim(0.0, 45.0)
# plt.ylim(0.0, 2.0)
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# plt.xticks([])
# plt.yticks([])
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=2))
# plt.tight_layout()
#
#
# plt.subplot(6, 1, 6)
# plt.plot(time4, pni61, color="#37A4F6", linewidth=8.0, linestyle='-')
# plt.plot(time2, pni6, color="#DB5F57", linewidth=8.0, linestyle='--')
# plt.xlim(0.0, 45.0)
# plt.ylim(0.0, 2.0)
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# plt.yticks([])
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=2))
# plt.tight_layout()
#
# plt.show()


'''损失函数'''
# time1 = [0.0, 10.0, 11.0, 18.2, 30.0, 31.0, 48.0]
# fix_num = [1, 5, 4, 8, 4, 4]
# # suofang = [1.5177934454122974, 1.2402532650316216, 1.0324298738316466, 1.3638973522819255, 1.4037427619481773, 1.1197429086663058, ]
#
# suofang = [10000, 10, 100, 100000, 100, 100]
# fig, ax = plt.subplots(figsize=(7.5, 7.5))
# for pha in range(len(fix_num)):
#     Los = np.empty([3, 0])
#     for fix in range(fix_num[pha]):
#         Loss = np.load(r"D:\pythonfile\GWO3\BQ\版本2\policy\data\NN_loss_Adv_phase=" + str(pha+1)
#                        + "_fix_num=" + str(fix) + "_KE=0.14_KR=0.008_KO=0.65.npy")
#         Los = np.hstack([Los, Loss])
#     Los *= suofang[pha]
#     # sf = 1.0  # suofang[pha]
#     # Los = (Los - np.min(Los)) / (np.max(Los) - np.min(Los)) * sf
#     plt.plot(np.linspace(time1[pha] + 0.1, time1[pha] + (fix_num[pha]+1) * 0.1, Los.shape[1]),
#              Los[0, :], linewidth=1.0, color='#DB5F57')
#
#     # Los[1, :] -= (Los[0, :] - Los[1, :]) * 1
#     # Los[2, :] += (-Los[0, :] + Los[2, :]) * 1
#     ax.fill_between(np.linspace(time1[pha] + 0.1, time1[pha] + (fix_num[pha]+1) * 0.1, Los.shape[1]),
#                     np.array(Los[1, :]), np.array(Los[2, :]), color='#DB5F57', alpha=0.5)
#
#     plt.plot(np.linspace(time1[pha] + (fix_num[pha]+1) * 0.1, time1[pha+1] + 0.1, 2),
#              np.linspace(Los[0, -1], Los[0, -1], 2), linewidth=1.0, color='#37A4F6', linestyle='--')
# plt.xlim(-0.5, 48.0)
# plt.ylim(-0.1, 6.0)
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
# # plt.grid()
# plt.tight_layout()
# plt.show()



'''导数'''
file = ["D:\pythonfile\GWO3\BQ\版本2\policy\data\ours_enemy_Adv_phase=", "D:\pythonfile\GWO3\BQ\版本2\policy\data\ours_NN_enemy_Adv_phase="]
data_ours = np.zeros([2000, 7])
data_ours_NN = np.zeros([2000, 7])
Len = 0
Len_NN = 0
for i in range(6):
    data = np.load(file[0] + str(i+1) + "_KE=0.14_KR=0.008_KO=0.65.npy", allow_pickle=True)
    data_ours[Len:Len+data.shape[0], :data.shape[1]] = data
    Len += data.shape[0]

    data1 = np.load(file[1] + str(i+1) + "_KE=0.14_KR=0.008_KO=0.65.npy", allow_pickle=True)
    data_ours_NN[Len_NN:Len_NN+data1.shape[0], :data1.shape[1]] = data1
    Len_NN += data1.shape[0]

    # for j in range(6):
    #     plt.subplot(6, 1, j + 1)
    #     plt.plot(data_ours_NN[:, 0], data_ours_NN[:, j + 1], color="#37A4F6", linewidth=6.0)
    # plt.show()

data_ours = data_ours[:Len, :]
data_ours_NN = data_ours_NN[:Len_NN, :]
for i in range(data_ours.shape[0]-1, 1):
    data_ours[i, 1:] = (data_ours[i, 1:]-data_ours[i-1, 1:]) / (data_ours[i, 0]-data_ours[i-1, 0])
for i in range(data_ours_NN.shape[0]-1, 1):
    data_ours_NN[i, 1:] = (data_ours_NN[i, 1:]-data_ours_NN[i-1, 1:]) / (data_ours_NN[i, 0]-data_ours_NN[i-1, 0])

data_ours = data_ours[1:, :]
data_ours_NN = data_ours_NN[1:, :]

fig, axes = plt.subplots(figsize=(7.5 * 2.32, 7.5))
for i in range(6):
    plt.subplot(6, 1, i+1)
    plt.plot(data_ours[:, 0], data_ours[:, i+1], color="#37A4F6", linewidth=8.0)
    plt.plot(data_ours_NN[:, 0], data_ours_NN[:, i+1], color="#DB5F57", linewidth=8.0, linestyle='--')

    plt.xlim(0.0, 48.0)
    plt.ylim(0.10, )
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=0)
    plt.yticks([])
    # plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=0))
    plt.tight_layout()
plt.show()