import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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



color = ['red', 'blue', 'green', 'black', 'orange', 'grey']
# plt.subplot(2, 1, 1)
# for i in range(1, 7):
#     data = np.load(
#         "D:\pythonfile\GWO3\BQ\版本2\policy\data\ours_NN_enemy_Adv_phase="+str(i)+"_KE=0.14_KR=0.008_KO=0.65.npy", allow_pickle=True)
#     for j in range(1, data.shape[1]):
#         plt.plot(data[:, 0], data[:, j], color=color[j-1])
#
# plt.subplot(2, 1, 2)
# for i in range(1, 7):
#     data = np.load(
#         "C:\\Users\王照宗\Desktop\data\policy1_enemy_Adv_phase="+str(i)+"_KE=0.14_KR=0.008_KO=0.65.npy", allow_pickle=True)
#     for j in range(1, data.shape[1]):
#         plt.plot(data[:, 0], data[:, j], color=color[j-1])
#
# plt.show()
#
# for i in range(1, 7):
#     data = np.load(
#         "D:\pythonfile\GWO3\BQ\版本2\policy\data\ours_NN_enemy_Adv_phase="+str(i)+"_KE=0.14_KR=0.008_KO=0.65.npy", allow_pickle=True)
#     for j in range(1, data.shape[1]):
#         plt.subplot(6, 1, j)
#         plt.plot(data[:, 0], data[:, j], color=color[j-1], linestyle='--')
#         plt.xlim(0, 50)
#         plt.ylim(0.08, 2.5)
#
#     data = np.load(
#         "C:\\Users\王照宗\Desktop\data\policy2_enemy_Adv_phase="+str(i)+"_KE=0.14_KR=0.008_KO=0.65.npy", allow_pickle=True)
#     for j in range(1, data.shape[1]):
#         plt.subplot(6, 1, j)
#         plt.plot(data[:, 0], data[:, j], color=color[j-1], linestyle=':')
#         plt.xlim(0, 50)
#         plt.ylim(0.08, 2.5)
#
#     data = np.load(
#         "C:\\Users\王照宗\Desktop\data\policy1_enemy_Adv_phase="+str(i)+"_KE=0.14_KR=0.008_KO=0.65.npy", allow_pickle=True)
#     for j in range(1, data.shape[1]):
#         plt.subplot(6, 1, j)
#         plt.plot(data[:, 0], data[:, j], color=color[j-1])
#         plt.xlim(0, 50)
#         plt.ylim(0.08, 2.5)
# plt.show()

# fig, ax = plt.subplots(figsize=(7.5 * 4, 7.5))
# for i in range(1, 4):
#     data = np.load(
#         "D:\pythonfile\GWO3\BQ\版本2\policy\data\ours_NN_enemy_Adv_phase=" + str(i) + "_KE=0.14_KR=0.008_KO=0.65.npy",
#         allow_pickle=True)
#     # ax.fill_between(data[:, 0], np.min(data[:, 1:], axis=1), np.max(data[:, 1:], axis=1), color='blue', alpha=0.1)
#     plt.plot(data[:, 0], np.sum(data[:, 1:], axis=1) / 4, color='blue')
#     # plt.plot(data[:, 0], np.min(data[:, 1:], axis=1), color='blue')
#     plt.plot(data[:, 0], np.max(data[:, 1:], axis=1), color='blue')
#
#     data = np.load(
#         "C:\\Users\王照宗\Desktop\data\policy1_enemy_Adv_phase="+str(i)+"_KE=0.14_KR=0.008_KO=0.65.npy",
#         allow_pickle=True)
#     # ax.fill_between(data[:, 0], np.min(data[:, 1:], axis=1), np.max(data[:, 1:], axis=1), color='orange', alpha=0.05)
#     plt.plot(data[:, 0], np.sum(data[:, 1:], axis=1) / 4, color='orange')
#     # plt.plot(data[:, 0], np.min(data[:, 1:], axis=1), color='orange')
#     plt.plot(data[:, 0], np.max(data[:, 1:], axis=1), color='orange')
#
#     data = np.load(
#         "C:\\Users\王照宗\Desktop\data\policy2_enemy_Adv_phase="+str(i)+"_KE=0.14_KR=0.008_KO=0.65.npy",
#         allow_pickle=True)
#     # ax.fill_between(data[:, 0], np.min(data[:, 1:], axis=1), np.max(data[:, 1:], axis=1), color='green', alpha=0.1)
#     plt.plot(data[:, 0], np.sum(data[:, 1:], axis=1) / 4, color='green')
#     # plt.plot(data[:, 0], np.min(data[:, 1:], axis=1), color='green')
#     plt.plot(data[:, 0], np.max(data[:, 1:], axis=1), color='green')
#     # data = np.load("C:\\Users\王照宗\Desktop\data\policy1_enemy_Adv_phase="+str(i)+"_KE=0.14_KR=0.008_KO=0.65.npy", allow_pickle=True)
#     # plt.plot(data[:, 0], np.sum(data[:, 1:], axis=1), color=color[i - 1], linestyle='--')
#     # data = np.load("C:\\Users\王照宗\Desktop\data\policy2_enemy_Adv_phase="+str(i)+"_KE=0.14_KR=0.008_KO=0.65.npy", allow_pickle=True)
#     # plt.plot(data[:, 0], np.sum(data[:, 1:], axis=1), color=color[i - 1], linestyle='-')
#
# for i in range(4, 7):
#     data = np.load(
#         "D:\pythonfile\GWO3\BQ\版本2\policy\data\ours_NN_enemy_Adv_phase=" + str(i) + "_KE=0.14_KR=0.008_KO=0.65.npy",
#         allow_pickle=True)
#     # ax.fill_between(data[:, 0], np.min(data[:, 1:], axis=1), np.max(data[:, 1:], axis=1), color='blue', alpha=0.1)
#     plt.plot(data[:, 0], np.sum(data[:, 1:], axis=1) / 6, color='blue')
#     # plt.plot(data[:, 0], np.min(data[:, 1:], axis=1), color='blue')
#     plt.plot(data[:, 0], np.max(data[:, 1:], axis=1), color='blue')
#
#     data = np.load(
#         "C:\\Users\王照宗\Desktop\data\policy1_enemy_Adv_phase="+str(i)+"_KE=0.14_KR=0.008_KO=0.65.npy",
#         allow_pickle=True)
#     # ax.fill_between(data[:, 0], np.min(data[:, 1:], axis=1), np.max(data[:, 1:], axis=1), color='orange', alpha=0.05)
#     plt.plot(data[:, 0], np.sum(data[:, 1:], axis=1) / 6, color='orange')
#     # plt.plot(data[:, 0], np.min(data[:, 1:], axis=1), color='orange')
#     plt.plot(data[:, 0], np.max(data[:, 1:], axis=1), color='orange')
#
#     data = np.load(
#         "C:\\Users\王照宗\Desktop\data\policy2_enemy_Adv_phase="+str(i)+"_KE=0.14_KR=0.008_KO=0.65.npy",
#         allow_pickle=True)
#     # ax.fill_between(data[:, 0], np.min(data[:, 1:], axis=1), np.max(data[:, 1:], axis=1), color='green', alpha=0.1)
#     plt.plot(data[:, 0], np.sum(data[:, 1:], axis=1) / 6, color='green')
#     # plt.plot(data[:, 0], np.min(data[:, 1:], axis=1), color='green')
#     plt.plot(data[:, 0], np.max(data[:, 1:], axis=1), color='green')
#     # data = np.load("C:\\Users\王照宗\Desktop\data\policy1_enemy_Adv_phase="+str(i)+"_KE=0.14_KR=0.008_KO=0.65.npy", allow_pickle=True)
#     # plt.plot(data[:, 0], np.sum(data[:, 1:], axis=1), color=color[i - 1], linestyle='--')
#     # data = np.load("C:\\Users\王照宗\Desktop\data\policy2_enemy_Adv_phase="+str(i)+"_KE=0.14_KR=0.008_KO=0.65.npy", allow_pickle=True)
#     # plt.plot(data[:, 0], np.sum(data[:, 1:], axis=1), color=color[i - 1], linestyle='-')
# plt.ylim(0.1, 2.5)
# plt.show()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'Times New Roman'

# file_name = [r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition2\policy1_KE=0.14_KR=0.008_KO=0.5_task_seq.xlsx',
#              r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition2\policy1_KE=0.16_KR=0.008_KO=0.5_task_seq.xlsx',
#              r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy1_KE=0.15_KR=0.005_KO=0.4_task_seq.xlsx',
#              r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy1_KE=0.17_KR=0.005_KO=0.4_task_seq.xlsx',
#              r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition4\policy1_KE=0.13_KR=0.011_KO=0.6_task_seq.xlsx',
#              r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition4\policy1_KE=0.15_KR=0.011_KO=0.6_task_seq.xlsx',]
#
# policy4_list = []
# Len3 = 0
# for j in range(len(file_name)):
#     data = pd.read_excel(file_name[j], sheet_name='Enemy')
#     data = np.array(data)
#     data = np.max(data[:, 1:], axis=1)
#     Len3 = max(Len3, data.shape[0])
#     policy4_list.append(data)
# data2 = np.zeros([Len3, len(policy4_list)])
# for j in range(len(file_name)):
#     data2[:policy4_list[j].shape[0], j] = policy4_list[j]
# fig, ax = plt.subplots(figsize=(15, 7.5))
# ax.fill_between(np.linspace(0, Len3*0.050, Len3), np.min(data2, axis=1), np.max(data2, axis=1), color='orange', alpha=0.05)
#
# file_name = [r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition2\policy5_KE=0.14_KR=0.008_KO=0.5_task_seq.xlsx',
#              r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition2\policy5_KE=0.16_KR=0.008_KO=0.5_task_seq.xlsx',
#              r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy5_KE=0.15_KR=0.005_KO=0.4_task_seq.xlsx',
#              r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy5_KE=0.17_KR=0.005_KO=0.4_task_seq.xlsx',
#              r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition4\policy5_KE=0.13_KR=0.011_KO=0.6_task_seq.xlsx',
#              r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition4\policy5_KE=0.15_KR=0.011_KO=0.6_task_seq.xlsx',]
#
# policy1_list = []
# Len1 = 0
# for j in range(len(file_name)):
#     data = pd.read_excel(file_name[j], sheet_name='Enemy')
#     data = np.array(data)
#     data = np.max(data[:, 1:], axis=1)
#     Len1 = max(Len1, data.shape[0])
#     policy1_list.append(data)
# data0 = np.zeros([Len1, len(policy1_list)])
# for j in range(len(file_name)):
#     data0[:policy1_list[j].shape[0], j] = policy1_list[j]
# ax.fill_between(np.linspace(0, Len1*0.05, Len1), np.min(data0, axis=1), np.max(data0, axis=1), color='green', alpha=0.1)
#
# file_name = [r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition2\policy3_KE=0.14_KR=0.008_KO=0.5_task_seq.xlsx',
#              r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition2\policy3_KE=0.16_KR=0.008_KO=0.5_task_seq.xlsx',
#              r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy3_KE=0.15_KR=0.005_KO=0.4_task_seq.xlsx',
#              r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy3_KE=0.17_KR=0.005_KO=0.4_task_seq.xlsx',
#              r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition4\policy3_KE=0.13_KR=0.011_KO=0.6_task_seq.xlsx',
#              r'C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition4\policy3_KE=0.15_KR=0.011_KO=0.6_task_seq.xlsx',]
# policy3_list = []
# Len2 = 0
# for j in range(len(file_name)):
#     data = pd.read_excel(file_name[j], sheet_name='Enemy')
#     data = np.array(data)
#     data = np.max(data[:, 1:], axis=1)
#     Len2 = max(Len2, data.shape[0])
#     policy3_list.append(data)
# data1 = np.zeros([Len2, len(policy3_list)])
# for j in range(len(file_name)):
#     data1[:policy3_list[j].shape[0], j] = policy3_list[j]
# ax.fill_between(np.linspace(0, Len2*0.05, Len2), np.min(data1, axis=1), np.max(data1, axis=1), color='blue', alpha=0.1)
#
# plt.plot(np.linspace(0, Len3*0.05, Len3), np.min(data2, axis=1), color='#E6A326', linewidth=2.0, label='Policy3 min/max')
# plt.plot(np.linspace(0, Len3*0.05, Len3), np.max(data2, axis=1), color='#E6A326', linewidth=2.0)
# plt.plot(np.linspace(0, Len3*0.05, Len3), np.sum(data2, axis=1) / data2.shape[1], color='#E6A326', linewidth=8.0, label='Policy3 average')
#
# plt.plot(np.linspace(0, Len1*0.05, Len1), np.min(data0, axis=1), color='#6EDB57', linewidth=2.0, label='Policy1 min/max')
# plt.plot(np.linspace(0, Len1*0.05, Len1), np.max(data0, axis=1), color='#6EDB57', linewidth=2.0)
# plt.plot(np.linspace(0, Len1*0.05, Len1), np.sum(data0, axis=1) / data0.shape[1], color='#6EDB57', linewidth=8.0, label='Policy1 average')
#
# plt.plot(np.linspace(0, Len2*0.05, Len2), np.min(data1, axis=1), color='#37A4F6', linewidth=2.0, label='Policy2 min/max')
# plt.plot(np.linspace(0, Len2*0.05, Len2), np.max(data1, axis=1), color='#37A4F6', linewidth=2.0)
# plt.plot(np.linspace(0, Len2*0.05, Len2), np.sum(data1, axis=1) / data1.shape[1], color='#37A4F6', linewidth=8.0, label='Policy2 average')
#
# plt.xlim(0, Len3*0.053)
# plt.ylim(0.05, 2.5)
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# # plt.xticks(fontsize=40)
# # plt.yticks(fontsize=40)
# # plt.xlabel("Time (s)", fontsize=40)
# # plt.ylabel("Max Anti-capability", fontsize=40)
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
# plt.grid()
# # plt.legend(fontsize=25, loc='upper right')
# plt.tight_layout()
# plt.show()

'''动态策略图'''
# color = ['#DB5F57', '#37A4F6', '#E6A326',  '#6EDB57']
# file = r"C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy3_KE=0.15_KR=0.005_KO=0.4_task_seq.xlsx"
# time = pd.read_excel(file, sheet_name='Agent_dic_time', header=None)
# time = np.array(time)
# agent_num = pd.read_excel(file, sheet_name='Agent_dic', header=None)
# agent_num = np.array(agent_num)
# ddd = []
# T = []
#
# for i in range(6):
#     min_time = np.max(time) / 6 * i
#     max_time = np.max(time) / 6 * (i+1)
#     condition = (time >= min_time) & (time < max_time)
#     time_index = np.where(condition)[0]
#     data = agent_num[time_index, :]
#     data_sum = np.sum(data, axis=0)
#     data_sum = data_sum / np.sum(data_sum)
#     ddd.append(data_sum)
#     T.append((max_time+min_time)/2*5.0)
#
# ddd = np.array(ddd)
# fig, ax = plt.subplots(figsize=(7.5*2.32, 7.5))
# for j in range(4):
#     ax.plot(T, ddd[:, j], color=color[j], linewidth=6.0, marker='*', markersize=35, label='opponent '+str(j+1))
# plt.xlim(0, np.max(time)*5.0)
# plt.ylim(-0.05, 0.9)
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# # plt.xticks(fontsize=40)
# # plt.yticks(fontsize=40)
# # plt.xlabel("Time (s)", fontsize=40)
# # plt.ylabel("Dynamic Policy", fontsize=40)
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.grid()
# # plt.legend(fontsize=25, loc='upper right', ncol=4)
# plt.tight_layout()
# plt.show()

'''敌方变化图'''
# fig, ax = plt.subplots(figsize=(15, 7.5))
# fig, ax = plt.subplots(figsize=(7.5*8.35/2.6, 7.5))
# file = r"C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy1_KE=0.15_KR=0.005_KO=0.4_task_seq.xlsx"
# rec_vel1 = pd.read_excel(file, sheet_name='Enemy_rec_vel')
# rec_vel1 = np.array(rec_vel1)
#
# file = r"C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy5_KE=0.15_KR=0.005_KO=0.4_task_seq.xlsx"
# rec_vel2 = pd.read_excel(file, sheet_name='Enemy_rec_vel')
# rec_vel2 = np.array(rec_vel2)
#
# file = r"C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy3_KE=0.15_KR=0.005_KO=0.4_task_seq.xlsx"
# rec_vel3 = pd.read_excel(file, sheet_name='Enemy_rec_vel')
# rec_vel3 = np.array(rec_vel3)
#
# ax.plot(rec_vel3[:, 0], np.sum(rec_vel3[:, 1:], axis=1), linestyle='--', color='#37A4F6', linewidth=8.0, label='policy3')
# ax.plot(rec_vel2[:, 0], np.sum(rec_vel2[:, 1:], axis=1), linestyle='--', color='#6EDB57', linewidth=8.0, label='policy2')
# ax.plot(rec_vel1[:, 0], np.sum(rec_vel1[:, 1:], axis=1), linestyle='--', color='#E6A326', linewidth=8.0, label='policy1')
#
#
# ax.scatter(0.1, 0.036, color='white', alpha=0.0)
#
# ax1 = ax.twinx()
# file = r"C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy1_KE=0.15_KR=0.005_KO=0.4_task_seq.xlsx"
# rec1 = pd.read_excel(file, sheet_name='Enemy_rec')
# rec1 = np.array(rec1)
#
# file = r"C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy5_KE=0.15_KR=0.005_KO=0.4_task_seq.xlsx"
# rec2 = pd.read_excel(file, sheet_name='Enemy_rec')
# rec2 = np.array(rec2)
#
# file = r"C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy3_KE=0.15_KR=0.005_KO=0.4_task_seq.xlsx"
# rec3 = pd.read_excel(file, sheet_name='Enemy_rec')
# rec3 = np.array(rec3)
#
# ax1.plot(rec3[:, 0], np.sum(rec3[:, 1:], axis=1), color='#37A4F6', linewidth=8.0, label='policy2')
# ax1.plot(rec2[:, 0], np.sum(rec2[:, 1:], axis=1), color='#6EDB57', linewidth=8.0, label='policy3')
# ax1.plot(rec1[:, 0], np.sum(rec1[:, 1:], axis=1), color='#E6A326', linewidth=8.0, label='policy1')
#
# ax1.plot(np.linspace(rec3[-1, 0], rec3[-1, 0]+100, 100), np.linspace(np.sum(rec3[:, 1:], axis=1)[-1], np.sum(rec3[:, 1:], axis=1)[-1], 100), color='#37A4F6', linewidth=8.0)
# ax1.plot(np.linspace(rec2[-1, 0], rec2[-1, 0]+100, 100), np.linspace(np.sum(rec2[:, 1:], axis=1)[-1], np.sum(rec2[:, 1:], axis=1)[-1], 100), color='#6EDB57', linewidth=8.0)
# ax1.plot(np.linspace(rec1[-1, 0], rec1[-1, 0]+100, 100), np.linspace(np.sum(rec1[:, 1:], axis=1)[-1], np.sum(rec1[:, 1:], axis=1)[-1], 100), color='#E6A326', linewidth=8.0)
#
# ax1.plot(np.linspace(rec3[-1, 0], rec3[-1, 0], 100), np.linspace(0, np.sum(rec3[:, 1:], axis=1)[-1], 100), linestyle='--', color='#37A4F6', linewidth=3.0)
# ax1.plot(np.linspace(rec2[-1, 0], rec2[-1, 0], 100), np.linspace(0, np.sum(rec2[:, 1:], axis=1)[-1], 100), linestyle='--', color='#6EDB57', linewidth=3.0)
# ax1.plot(np.linspace(rec1[-1, 0], rec1[-1, 0], 100), np.linspace(0, np.sum(rec1[:, 1:], axis=1)[-1], 100), linestyle='--', color='#E6A326', linewidth=3.0)
#
# plt.xlim(0, max(max(rec3[-1, 0], rec2[-1, 0]), rec1[-1, 0])+20)
# plt.ylim(0, 7.0)
# ax.set_ylim(0.0042, )
# # ax1.xlim(0, 280)
# # ax1.ylim(0,)
# # ax.xticks(fontsize=0)
# # ax.yticks(fontsize=0)
# # ax1.xticks(fontsize=0)
# # ax1.yticks(fontsize=0)
# # plt.xlabel("Time (s)", fontsize=40)
# # plt.ylabel("Dynamic Policy", fontsize=40)
# plt.xticks(fontsize=0)
# ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
# ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))
# plt.grid()
# # plt.legend(fontsize=25, loc='upper right')
# plt.tight_layout()
# plt.show()

'''出动次数'''
# file = r"C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy3_KE=0.15_KR=0.005_KO=0.4_task_seq.xlsx"
# data = pd.read_excel(file, header=None, sheet_name='Agent_dic_time')
# time = np.array(data)
# data = pd.read_excel(file, header=None, sheet_name='Agent_dic')
# agent_num = np.sum(np.array(data), axis=1)
# T = [time[0]]
# A = [agent_num[0]]
# for i in range(time.shape[0]-1):
#     T.append(time[i+1]*5.0)
#     A.append(A[-1])
#     T.append(time[i+1]*5.0)
#     A.append(agent_num[i+1]+A[-1])
# # fig, ax = plt.subplots(figsize=(7.5/2.6*8.3, 7.5))
# fig, ax = plt.subplots(figsize=(15, 7.5))
# plt.plot(T, A, color='#37A4F6', linewidth=18.0)
# plt.plot(np.linspace(T[-1], T[-1]+100, 10), np.linspace(A[-1], A[-1], 10), color='#37A4F6', linewidth=18.0)
# plt.plot(np.linspace(T[-1], T[-1], 100), np.linspace(0, A[-1], 100), color='#37A4F6', linewidth=7.0, linestyle='--')
#
# file = r"C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy5_KE=0.15_KR=0.005_KO=0.4_task_seq.xlsx"
# data = pd.read_excel(file, header=None, sheet_name='Agent_dic_time')
# time = np.array(data)
# data = pd.read_excel(file, header=None, sheet_name='Agent_dic')
# agent_num = np.sum(np.array(data), axis=1)
# T = [time[0]]
# A = [agent_num[0]]
# for i in range(time.shape[0]-1):
#     T.append(time[i+1]*5.0)
#     A.append(A[-1])
#     T.append(time[i+1]*5.0)
#     A.append(agent_num[i+1]+A[-1])
# plt.plot(T, A, color='#6EDB57', linewidth=18.0)
# plt.plot(np.linspace(T[-1], T[-1]+100, 10), np.linspace(A[-1], A[-1], 10), color='#6EDB57', linewidth=18.0)
# plt.plot(np.linspace(T[-1], T[-1], 100), np.linspace(0, A[-1], 100), color='#6EDB57', linewidth=7.0, linestyle='--')
#
# file = r"C:\Users\王照宗\Desktop\Adversarial_process\P=4\condition3\policy1_KE=0.15_KR=0.005_KO=0.4_task_seq.xlsx"
# data = pd.read_excel(file, header=None, sheet_name='Agent_dic_time')
# time = np.array(data)
# data = pd.read_excel(file, header=None, sheet_name='Agent_dic')
# agent_num = np.sum(np.array(data), axis=1)
# T = [time[0]]
# A = [agent_num[0]]
# for i in range(time.shape[0]-1):
#     T.append(time[i+1]*5.0)
#     A.append(A[-1])
#     T.append(time[i+1]*5.0)
#     A.append(agent_num[i+1]+A[-1])
# plt.plot(T, A, color='#E6A326', linewidth=18.0)
# plt.plot(np.linspace(T[-1], T[-1]+100, 10), np.linspace(A[-1], A[-1], 10), color='#E6A326', linewidth=18.0)
# plt.plot(np.linspace(T[-1], T[-1], 100), np.linspace(0, A[-1], 100), color='#E6A326', linewidth=7.0, linestyle='--')
#
# # plt.xscale('log')
# plt.xlim(0, np.max(time)*5.0+20)
# plt.ylim(0,)
# plt.xticks(fontsize=0)
# plt.yticks(fontsize=0)
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
# # plt.grid()
# plt.tight_layout()
# plt.show()



# fig, ax = plt.subplots(figsize=(10, 4))
#
# for i in range(enemy.shape[1]-1):
#     ax.plot(enemy[:, 0], enemy[:, i+1], linewidth=1, c=color[i])
#
# for spine in ax.spines.values():
#     spine.set_visible(False)  # 隐藏坐标轴边框
# ax.xaxis.set_visible(False)  # 隐藏x轴
# ax.yaxis.set_visible(False)  # 隐藏y轴
# ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False,
#                labelleft=False)
# fig.patch.set_alpha(0)  # 设置整个图形的背景为透明
# plt.savefig('line_plot_transparent.png', dpi=900, transparent=True, bbox_inches='tight', pad_inches=0)


# enemy = pd.read_excel(r'F:\仿真\test\Assets\Data\policy1_KE=0.16_task_seq.xlsx', sheet_name='Enemy')
# enemy = np.array(enemy)
# for i in range(enemy.shape[1]-1):
#     plt.plot(enemy[:, 0], enemy[:, i+1], linewidth=1.5, c=color[i])
# plt.fill_between(enemy[:, 0], np.min(enemy[:, 1:], axis=1), np.max(enemy[:, 1:], axis=1), color=color[-1])
#
# plt.grid()
# plt.show()

# fig, ax = plt.subplots(figsize=(7.5*8.35/2.6, 7.5))
# policy1 = np.array([287.5382534,287.5382534,281.257119,280.8795184,333.4400886,333.4400886,392.1953815,392.7804236,393.1748514,414.59607,414.0222939,414.5686396,])
# policy2 = np.array([274.6096378,274.9442779,273.1325856,272.7485113,305.6995292,304.601671,290.0735739,291.532768,289.5823047,336.877792,335.8415767,335.3223732,])
# policy3 = np.array([240.3410489,226.8596523,247.4563452,234.3598955,260.1513993,226.3649105,255.5,236.9062685,214.7206443,304.6476507,295.8729366,267.4162301,])
# width = 1.0
# for i in range(policy1.shape[0]):
#     plt.bar(5*i, policy1[i], width=width, color="#E6A326")
#     plt.bar(5*i+width*1.2, policy2[i], width=width, color="#6EDB57")
#     plt.bar(5*i+2*width*1.2, policy3[i], width=width, color="#37A4F6")
# plt.xlim(-1, (policy1.shape[0]-1)*5+2*width*1.2+1)
# ax.set_xticks([])
# plt.yticks(fontsize=0.0)
# # ax.set_yticks([])
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
# plt.tight_layout()
# plt.show()


Time = [[48.16583, 37.51641, 52.10512, 35.03315],
        [44.91476, 36.11035, 36.06258, 29.25078],
        [37.16299, 29.82377, 31.18842, 25.96684],
        [37.69573, 29.83205, 31.05962, 26.00000]]

width = 1.0
fig, ax = plt.subplots(figsize=(7.5*2.32, 7.5))

for i in range(4):
    ax.bar(6 * i, Time[0][i], width=width, color="#E6A326")
    ax.bar(6 * i + width*1.2, Time[1][i], width=width, color="#6EDB57")
    ax.bar(6 * i + width*2.4, Time[2][i], width=width, color="#37A4F6")
    ax.bar(6 * i + width*3.6, Time[3][i]*1.03, width=width, color="#37A4F6", edgecolor="black", linewidth=2, hatch='/')


ax.set_xticks([])
plt.ylim(0, 60)
plt.yticks(fontsize=0.0)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
plt.tight_layout()
plt.show()
sys.exit()



















P4 = np.array([[274.6096378,274.9442779,273.1325856,272.7485113,305.6995292,304.601671],
               [262.0218803,262.0218803,262.0218803,262.0218803,281.6379063,281.637906],
               [240.3410489,226.8596523,247.4563452,234.3598955,260.1513993,226.364910]])
P6 = np.array([[291.532768,289.5823047,290.9900586,335.8415767,335.3223732,333.9014418],
               [248.609023,248.6078106,248.5409888,297.1412912,297.1412912,297.1412912],
               [236.906269,214.7206443,198.5623434,295.8729366,267.4162301,247.0726104]])
P8 = np.array([[282.5548387,282.855481,282.6878493,424.7562893,423.6297505,422.9228242],
               [219.4456414,219.6006533,218.9414012,341.2503295,341.2503295,341.3795282],
               [192.5353608,189.0409318,184.3473202,326.468248,309.3924715,307.7448234]])
Lis = [P4, P6, P8]

width = 1.0
fig, ax = plt.subplots(figsize=(7.5*2.32, 7.5))
for i in range(3):
    print(np.sum(Lis[i][0]) / 6)
    print(np.sum(Lis[i][1]) / 6)
    print(np.sum(Lis[i][2]) / 6)
    print("")

    ax.bar(5 * i, np.sum(Lis[i][0])/6, width=width, color="#E6A326")
    ax.plot(np.linspace(5 * i - width * 0.3, 5 * i + width * 0.3, 20),
            np.linspace(np.max(Lis[i][0]), np.max(Lis[i][0]), 20), color='black', linewidth=3.0)
    ax.plot(np.linspace(5 * i - width * 0.3, 5 * i + width * 0.3, 20),
            np.linspace(np.min(Lis[i][0]), np.min(Lis[i][0]), 20), color='black', linewidth=3.0)
    ax.plot(np.linspace(5 * i, 5 * i, 20),
            np.linspace(np.min(Lis[i][0]), np.max(Lis[i][0]), 20), color='black', linewidth=3.0)

    ax.bar(5 * i + width*1.2, np.sum(Lis[i][1])/6, width=width, color="#6EDB57")
    ax.plot(np.linspace(5 * i + width*1.2 - width * 0.3, 5 * i + width*1.2 + width * 0.3, 20),
            np.linspace(np.max(Lis[i][1]), np.max(Lis[i][1]), 20), color='black', linewidth=3.0)
    ax.plot(np.linspace(5 * i + width*1.2 - width * 0.3, 5 * i + width*1.2 + width * 0.3, 20),
            np.linspace(np.min(Lis[i][1]), np.min(Lis[i][1]), 20), color='black', linewidth=3.0)
    ax.plot(np.linspace(5 * i + width*1.2, 5 * i + width*1.2, 20),
            np.linspace(np.min(Lis[i][1]), np.max(Lis[i][1]), 20), color='black', linewidth=3.0)

    ax.bar(5 * i + width*2.4, np.sum(Lis[i][2])/6, width=width, color="#37A4F6")
    ax.plot(np.linspace(5 * i + width*2.4 - width * 0.3, 5 * i + width*2.4 + width * 0.3, 20),
            np.linspace(np.max(Lis[i][2]), np.max(Lis[i][2]), 20), color='black', linewidth=3.0)
    ax.plot(np.linspace(5 * i + width*2.4 - width * 0.3, 5 * i + width*2.4 + width * 0.3, 20),
            np.linspace(np.min(Lis[i][2]), np.min(Lis[i][2]), 20), color='black', linewidth=3.0)
    ax.plot(np.linspace(5 * i + width*2.4, 5 * i + width*2.4, 20),
            np.linspace(np.min(Lis[i][2]), np.max(Lis[i][2]), 20), color='black', linewidth=3.0)

ax1 = ax.twinx()
ax1.plot(np.array([1.2, 6.2, 11.2]),
            np.array([0.84159159, 0.77805986, 0.712241269]), color="#E6A326", marker='p', linewidth=6.0, markersize=35)
ax1.plot(np.array([1.2, 6.2, 11.2]),
            np.array([0.89888117, 0.895112975, 0.892531352]), color="#6EDB57", marker='h', linewidth=6.0, markersize=35)


ax.set_ylim(0, 700)
ax1.set_ylim(0, 1.1)
ax.set_xticks([])
ax1.set_xticks([])
# ax.set_yticks([])
ax1.set_yticks([])
# plt.ylim(0, 550)
plt.yticks(fontsize=0.0)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
plt.tight_layout()
plt.show()

'''
Agent_number = 100
P = 6
KE = 0.09
KR = 0.005
KO = 0.4
for ii in range(2, 3):
    a = np.load("D:\pythonfile\GWO1\BQ\版本2\policy\data\policy"+str(ii+1)+"_confirmed_KE=" + str(KE) + '_KR=' + str(KR) + '_KO=' + str(KO) + ".npy", allow_pickle=True)
    a[3, :] *= Agent_number

    b = np.load("D:\pythonfile\GWO1\BQ\版本2\policy\data\policy"+str(ii+1)+"_confirmed_index_KE=" + str(KE) + '_KR=' + str(KR) + '_KO=' + str(KO) + ".npy", allow_pickle=True)

    data2 = np.ones([Agent_number, a.shape[1]]).astype(int) * P
    for index in range(b.shape[0]):
        for jndex in range(len(b[index])):
            data2[b[index][jndex], np.where(data2[b[index][jndex], :] == P)[0][0]] = int(a[0, index] - 1)
    data2 = pd.DataFrame(data2)

    c = np.load("D:\pythonfile\GWO1\BQ\版本2\policy\data\policy"+str(ii+1)+"_enemy_KE=" + str(KE) + '_KR=' + str(KR) + '_KO=' + str(KO) + ".npy", allow_pickle=True)

    d = np.load("D:\pythonfile\GWO1\BQ\版本2\policy\data\policy"+str(ii+1)+"_enemy_rec_KE=" + str(KE) + '_KR=' + str(KR) + '_KO=' + str(KO) + ".npy", allow_pickle=True)

    e = np.load("D:\pythonfile\GWO1\BQ\版本2\policy\data\policy"+str(ii+1)+"_enemy_exc_KE=" + str(KE) + '_KR=' + str(KR) + '_KO=' + str(KO) + ".npy", allow_pickle=True)

    f = np.load("D:\pythonfile\GWO1\BQ\版本2\policy\data\policy"+str(ii+1)+"_agent_dic_mak=" + str(KE) + '_KR=' + str(KR) + '_KO=' + str(KO) + ".npy", allow_pickle=True)

    g = np.load("D:\pythonfile\GWO1\BQ\版本2\policy\data\policy"+str(ii+1)+"_agent_dic_mak_time=" + str(KE) + '_KR=' + str(KR) + '_KO=' + str(KO) + ".npy", allow_pickle=True)

    h = np.load("D:\pythonfile\GWO1\BQ\版本2\policy\data\policy"+str(ii+1)+"_enemy_rec_vel_KE=" + str(KE) + '_KR=' + str(KR) + '_KO=' + str(KO) + ".npy", allow_pickle=True)

    i = np.load("D:\pythonfile\GWO1\BQ\版本2\policy\data\policy"+str(ii+1)+"_enemy_exc_vel_KE=" + str(KE) + '_KR=' + str(KR) + '_KO=' + str(KO) + ".npy", allow_pickle=True)


    Lis = [0]
    ind = 0
    while(True):
        index = np.where(c[:, 0] <= 0.05 * (ind+1))[0][-1]
        if(Lis[-1] != index):
            Lis.append(index)
        else:
            break
        ind += 1
    print("!@312")
    c = c[Lis, :]
    d = d[Lis, :]
    e = e[Lis, :]
    h = h[Lis, :]
    i = i[Lis, :]
    # c = c[0:c.shape[0]:10, :]
    # d = d[0:d.shape[0]:10, :]
    # e = e[0:e.shape[0]:10, :]
    # h = d[0:h.shape[0]:10, :]
    # i = e[0:i.shape[0]:10, :]

    data0 = pd.DataFrame(a)
    data1 = pd.DataFrame(b)
    data3 = pd.DataFrame(c)
    data4 = pd.DataFrame(d)
    data5 = pd.DataFrame(e)
    data6 = pd.DataFrame(f)
    data7 = pd.DataFrame(g)
    data8 = pd.DataFrame(h)
    data9 = pd.DataFrame(i)

    writer = pd.ExcelWriter(r'F:\仿真\test\Assets\Data1\policy'+str(ii+1)+'_KE=' + str(KE) + '_KR=' + str(KR) + '_KO=' + str(KO) + '_task_seq.xlsx')
    data0.to_excel(writer, 'Task_seq1', float_format='%.15f', header=None, index=False)
    data1.to_excel(writer, 'Task_seq2', float_format='int', header=None, index=False)
    data2.to_excel(writer, 'Task_seq3', float_format='int', header=None, index=False)
    data3.to_excel(writer, 'Enemy', float_format='%.15f', header=None, index=False)
    data4.to_excel(writer, 'Enemy_rec', float_format='%.15f', header=None, index=False)
    data5.to_excel(writer, 'Enemy_exc', float_format='%.15f', header=None, index=False)
    data6.to_excel(writer, 'Agent_dic', float_format='%.15f', header=None, index=False)
    data7.to_excel(writer, 'Agent_dic_time', float_format='%.15f', header=None, index=False)
    data8.to_excel(writer, 'Enemy_rec_vel', float_format='%.15f', header=None, index=False)
    data9.to_excel(writer, 'Enemy_exc_vel', float_format='%.15f', header=None, index=False)
    writer.save()
    writer.close()
    '''