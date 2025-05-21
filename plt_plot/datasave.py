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
import pickle
sys.path.append("..")
import parameter
import os

current_script_path = os.path.abspath(sys.argv[0])
current_script_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_script_dir)
file = [parent_dir + r"\data\policy1\policy1_", parent_dir + r"\data\policy2\policy2_",
        parent_dir + r"\data\ours\ours_", parent_dir + r"\data\ours_NN\ours_NN_"]

P = max(parameter.P)
KE = parameter.KE
KR = parameter.KR
KO = parameter.KO
file_name = ["enemy", "enemy_exc", "enemy_rec", "enemy_exc_vel", "enemy_rec_vel",
             "enemy_agent_dic_mak", "enemy_agent_dic_mak_time"]
for i in range(len(file)):
    Time = []
    data_enemy = np.zeros([10000, P+1])
    data_enemy_exc = np.zeros([10000, P**2+1])
    data_enemy_rec = np.zeros([10000, P+1])
    data_enemy_exc_vel = np.zeros([10000, P**2+1])
    data_enemy_rec_vel = np.zeros([10000, P+1])
    data_enemy_agent1 = np.zeros([10000, P])
    data_enemy_agent2 = np.zeros([10000])
    Len = 0
    Len1 = 0

    data_enemy_exc1 = np.zeros([P**2])
    data_enemy_rec1 = np.zeros([P])

    for j in range(1, len(parameter.P)):
        F = file[i] + file_name[0] + "_Adv_phase=" + str(j) + "_KE=" + str(KE) + "_KR=" + str(KR) + "_KO=" + str(KO) + ".npy"
        try:
            data = np.load(F, allow_pickle=True)
            data_enemy[Len:data.shape[0] + Len, :1 + parameter.P[j]] = data
        except FileNotFoundError:
            data_enemy = data_enemy

        F = file[i] + file_name[1] + "_Adv_phase=" + str(j) + "_KE=" + str(KE) + "_KR=" + str(KR) + "_KO=" + str(KO) + ".npy"
        try:
            data = np.load(F, allow_pickle=True)
            data_enemy_exc[Len:data.shape[0] + Len, 0] = data[:, 0]
            for jj in range(parameter.P[j]):
                data_enemy_exc[Len:data.shape[0] + Len, (1+P*jj):(1+P*jj+parameter.P[j])] = data[:, (1+parameter.P[j]*jj):(1+parameter.P[j]*(jj+1))]
            data_enemy_exc[Len:data.shape[0] + Len, 1:] += data_enemy_exc1
            data_enemy_exc1 = data_enemy_exc[data.shape[0] + Len - 1, 1:]
        except FileNotFoundError:
            data_enemy_exc = data_enemy_exc
            data_enemy_exc1 = data_enemy_exc1

        F = file[i] + file_name[2] + "_Adv_phase=" + str(j) + "_KE=" + str(KE) + "_KR=" + str(KR) + "_KO=" + str(KO) + ".npy"
        try:
            data = np.load(F, allow_pickle=True)
            data_enemy_rec[Len:data.shape[0] + Len, :(1 + parameter.P[j])] = data
            data_enemy_rec[Len:data.shape[0] + Len, 1:] += data_enemy_rec1
            data_enemy_rec1 = data_enemy_rec[data.shape[0] + Len - 1, 1:]
        except FileNotFoundError:
            data_enemy_rec = data_enemy_rec
            data_enemy_rec1 = data_enemy_rec1

        F = file[i] + file_name[3] + "_Adv_phase=" + str(j) + "_KE=" + str(KE) + "_KR=" + str(KR) + "_KO=" + str(KO) + ".npy"
        try:
            data = np.load(F, allow_pickle=True)
            data_enemy_exc_vel[Len:data.shape[0] + Len, 0] = data[:, 0]
            for jj in range(parameter.P[j]):
                data_enemy_exc_vel[Len:data.shape[0] + Len, (1 + P * jj):(1 + P * jj + parameter.P[j])] = data[:, (1 +parameter.P[j] * jj):(1 +parameter.P[j] * (jj + 1))]
        except FileNotFoundError:
            data_enemy_exc_vel = data_enemy_exc_vel

        F = file[i] + file_name[4] + "_Adv_phase=" + str(j) + "_KE=" + str(KE) + "_KR=" + str(KR) + "_KO=" + str(KO) + ".npy"
        try:
            data = np.load(F, allow_pickle=True)
            data_enemy_rec_vel[Len:data.shape[0] + Len, :(1 + parameter.P[j])] = data
        except FileNotFoundError:
            data_enemy_rec_vel = data_enemy_rec_vel

        F = file[i] + file_name[5] + "_Adv_phase=" + str(j) + "_KE=" + str(KE) + "_KR=" + str(KR) + "_KO=" + str(KO) + ".npy"
        try:
            data1 = np.load(F, allow_pickle=True)
            data_enemy_agent1[Len1:data1.shape[0] + Len1, :parameter.P[j]] = data1
        except FileNotFoundError:
            data_enemy_agent1 = data_enemy_agent1

        F = file[i] + file_name[6] + "_Adv_phase=" + str(j) + "_KE=" + str(KE) + "_KR=" + str(KR) + "_KO=" + str(KO) + ".npy"
        try:
            data1 = np.load(F, allow_pickle=True)
            data_enemy_agent2[Len1:data1.shape[0] + Len1] = data1
        except FileNotFoundError:
            data_enemy_agent2 = data_enemy_agent2

        Len += data.shape[0]
        Len1 += data1.shape[0]

    writer = pd.ExcelWriter(file[i] + "task_seq.xlsx")

    data = pd.DataFrame(data_enemy[:Len, :])
    data.to_excel(writer, 'Enemy', float_format='%.15f', header=None, index=False)

    data = pd.DataFrame(data_enemy_exc[:Len, :])
    data.to_excel(writer, 'Enemy_exc', float_format='%.15f', header=None, index=False)

    data = pd.DataFrame(data_enemy_rec[:Len, :])
    data.to_excel(writer, 'Enemy_rec', float_format='%.15f', header=None, index=False)

    data = pd.DataFrame(data_enemy_exc_vel[:Len, :])
    data.to_excel(writer, 'Enemy_exc_vel', float_format='%.15f', header=None, index=False)

    data = pd.DataFrame(data_enemy_rec_vel[:Len, :])
    data.to_excel(writer, 'Enemy_rec_vel', float_format='%.15f', header=None, index=False)

    data = pd.DataFrame(data_enemy_agent1[:Len1, :])
    data.to_excel(writer, 'Agent_dic', float_format='%.15f', header=None, index=False)

    data = pd.DataFrame(data_enemy_agent2[:Len1])
    data.to_excel(writer, 'Agent_dic_time', float_format='%.15f', header=None, index=False)

    if(i == 3):
        for index in range(100):
            F = file[i] + "confirmed_Adv_phase=" + str(len(parameter.P)-1-index) + "_KE=" + str(KE) + "_KR=" + str(
                KR) + "_KO=" + str(KO) + ".npy"
            try:
                data = np.load(F, allow_pickle=True)
                index = np.where([data[0, :] == 0.0])[-1][0]
                data[3, :] *= parameter.Agent_number
                data = pd.DataFrame(data[:, :index])
                data.to_excel(writer, 'Task_seq1', float_format='%.15f', header=None, index=False)
                break
            except FileNotFoundError:
                None

        for index in range(100):
            F = file[i] + "confirmed_index_Adv_phase=" + str(len(parameter.P)-1-index) + "_KE=" + str(KE) + "_KR=" + str(
                KR) + "_KO=" + str(KO) + ".pkl"
            try:
                with open(F, 'rb') as file:
                    data = pickle.load(file)
                data = np.array(data)
                data = pd.DataFrame(data[:index])
                data.to_excel(writer, 'Task_seq2', float_format='%.15f', header=None, index=False)
                break
            except FileNotFoundError:
                None

    else:
        data = np.load(file[i] + "confirmed" + "_KE=" + str(KE) + "_KR=" + str(KR) + "_KO=" + str(KO) + ".npy", allow_pickle=True)
        data = pd.DataFrame(data)
        data.to_excel(writer, 'Task_seq1', float_format='%.15f', header=None, index=False)

        data = np.load(file[i] + "confirmed_index" + "_KE=" + str(KE) + "_KR=" + str(KR) + "_KO=" + str(KO) + ".npy", allow_pickle=True)
        data = pd.DataFrame(data)
        data.to_excel(writer, 'Task_seq2', float_format='%.15f', header=None, index=False)

    writer.save()
    writer.close()

writer = pd.ExcelWriter(r'C:\Users\王照宗\Desktop\Adversarial_process\Dynamic1\flight_point.xlsx', mode='a', if_sheet_exists='replace',
                        engine='openpyxl')
Time = parameter.Time[0]
Bij = parameter.Bij[0]
data = np.hstack([np.array([max(parameter.P), 5.0]), Time])
data = pd.DataFrame(data)
data.to_excel(writer, 'P', float_format='%.15f', header=None, index=False)
Bij = pd.DataFrame(Bij)
Bij.to_excel(writer, 'P1', float_format='%.15f', header=None, index=False)
writer.save()
writer.close()