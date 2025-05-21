import numpy as np
import pandas as pd
import os
import sys
from openpyxl import Workbook, load_workbook
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
import parameter
import NN.NN_model as NN_model

model_path = sys.path[0] + r"\trained_NN\Adv_phase="

# class Model_fine_tuning:
#     @staticmethod
#     def print_parameters_in_model(model):
#         print("Parameters in IE network:")
#         for name, param in model.Son_swapnet.fc1.named_parameters():
#             print(param.data)
#
#         print("Parameters in BC network:")
#         for name, param in model.Son_swapnet.fc2.named_parameters():
#             print(param.data)
#
#         print("Parameters in TD network:")
#         for name, param in model.Son_swapnet.fc3.named_parameters():
#             print(param.data)
#
#         print("Parameters in sub-recovery network:")
#         for name, param in model.Son_recovernet.fc.named_parameters():
#             print(param.data)
#         print(" ")
#
#     # @staticmethod
#     # def load_ae(retrain, P, P_last_phase, phase):
#     #     if(retrain):
#     #         if(phase == 0):
#     #             ae1 = NN_model.DiagonalLinear(channels=P)
#     #             ae2 = NN_model.DiagonalLinear(channels=P)
#     #             ae1.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae1_Adv_phase=" + str(phase) + ".pkl"))
#     #             ae2.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae2_Adv_phase=" + str(phase) + ".pkl"))
#     #         else:
#     #             ae1_old = NN_model.DiagonalLinear(channels=P_last_phase)
#     #             ae2_old = NN_model.DiagonalLinear(channels=P_last_phase)
#     #             ae1_old.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae1_Adv_phase=" + str(phase-1) + ".pkl"))
#     #             ae2_old.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae2_Adv_phase=" + str(phase-1) + ".pkl"))
#     #             ae1 = NN_model.DiagonalLinear(channels=P)
#     #             ae2 = NN_model.DiagonalLinear(channels=P)
#     #
#     #             if P >= P_last_phase:
#     #                 print(phase)
#     #                 ae1.diagonal[:P_last_phase] = ae1_old.diagonal
#     #                 ae2.diagonal[:P_last_phase] = ae2_old.diagonal
#     #             else:
#     #                 change_index = [parameter.opponents_index[phase-1].index(x) for x in
#     #                                 parameter.opponents_index[phase]]
#     #                 ae1.diagonal = ae1_old.diagonal[change_index]
#     #                 ae2.diagonal = ae2_old.diagonal[change_index]
#     #     else:
#     #         ae1 = NN_model.DiagonalLinear(channels=P)
#     #         ae2 = NN_model.DiagonalLinear(channels=P)
#     #         ae1.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae1_Adv_phase=" + str(phase) + ".pkl"))
#     #         ae2.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae2_Adv_phase=" + str(phase) + ".pkl"))
#     #     return ae1, ae2
#
#     @staticmethod
#     def load_ae(P, phase):
#         ae1 = NN_model.DiagonalLinear(channels=P)
#         ae2 = NN_model.DiagonalLinear(channels=P)
#         ae1.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae1_Adv_phase=" + str(phase) + ".pkl"))
#         ae2.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae2_Adv_phase=" + str(phase) + ".pkl"))
#         return ae1, ae2
#
#     @staticmethod
#     def old_transfer_new_ae(retrain, P, P_last_phase, phase):
#         if (retrain):
#             if (phase == 0):
#                 ae1, ae2 = Model_fine_tuning.load_ae(P, phase)
#             else:
#                 ae1_old, ae2_old = Model_fine_tuning.load_ae(P_last_phase, phase-1)
#                 ae1 = NN_model.DiagonalLinear(channels=P)
#                 ae2 = NN_model.DiagonalLinear(channels=P)
#
#                 if P >= P_last_phase:
#                     # with torch.no_grad():
#                     #     ae1.diagonal[:P_last_phase] = ae1_old.diagonal
#                     #     ae2.diagonal[:P_last_phase] = ae2_old.diagonal
#                     # 保留旧模型参数，扩展新的
#                     new_diag1 = torch.ones(P, device=ae1_old.diagonal.device)
#                     new_diag2 = torch.ones(P, device=ae2_old.diagonal.device)
#                     new_diag1[:P_last_phase] = ae1_old.diagonal
#                     new_diag2[:P_last_phase] = ae2_old.diagonal
#                     ae1.diagonal = nn.Parameter(new_diag1)
#                     ae2.diagonal = nn.Parameter(new_diag2)
#                 else:
#                     preserve_index = [parameter.opponents_index[phase - 1].index(x) for x in
#                                       parameter.opponents_index[phase]]
#                     # with torch.no_grad():
#                     #     ae1.diagonal = ae1_old.diagonal[preserve_index]
#                     #     ae2.diagonal = ae2_old.diagonal[preserve_index]
#                     new_diag1 = ae1_old.diagonal[preserve_index]
#                     new_diag2 = ae2_old.diagonal[preserve_index]
#                     ae1.diagonal = nn.Parameter(new_diag1)
#                     ae2.diagonal = nn.Parameter(new_diag2)
#         else:
#             ae1, ae2 = Model_fine_tuning.load_ae(P, phase)
#         return ae1, ae2
#
#     @staticmethod
#     def load_model(P, order, phase):
#         model = NN_model.Opponent_model_explicit(channels=P, order=order)
#         model.load_state_dict(torch.load(model_path + str(phase) + ".pkl"))  # 执行方对任务方的建模
#         return model
#
#     @staticmethod
#     def old_transfer_new(retrain, P, P_last_phase, phase):
#         opponents_number_change = P - P_last_phase
#         if (not retrain):
#             print("Phase " + str(phase) + " opponent model being used")
#             model = Model_fine_tuning.load_model(P, order, phase)
#             Model_fine_tuning.print_parameters_in_model(model)
#         else:
#             print("Phase " + str(phase-1) + " opponent model being transferred")
#             if (opponents_number_change == 0):
#                 model = Model_fine_tuning.load_model(P, order, phase-1)
#             else:
#                 model_old = Model_fine_tuning.load_model(P_last_phase, order, phase-1)
#                 model = NN_model.Opponent_model_explicit(channels=P, order=order)
#
#                 # Transfer parameters in old network to new network
#                 with torch.no_grad():
#                     # Recovery network transfer
#                     model.Son_recovernet.fc.load_state_dict(model_old.Son_recovernet.fc.state_dict())
#
#                     # IE network transfer
#                     indices = [parameter.opponents_index[phase-1].index(x) for x in parameter.opponents_index[phase]]
#                     if (opponents_number_change < 0):
#                         model.Son_swapnet.fc1.diagonal.data.copy_(model_old.Son_swapnet.fc1.diagonal.data[indices])
#                     else:
#                         model.Son_swapnet.fc1.diagonal.data[:P_last_phase].copy_(model_old.Son_swapnet.fc1.diagonal.data)
#                         model.Son_swapnet.fc1.diagonal.data[P_last_phase:] = model_old.Son_swapnet.fc1.diagonal.data.sum().item() / P_last_phase
#
#                     # BC network transfer
#                     model.Son_swapnet.fc2.load_state_dict(model_old.Son_swapnet.fc2.state_dict())
#
#                     # TD network transfer
#                     for parameters in model_old.Son_swapnet.fc3.parameters():  # 打印出参数矩阵及值
#                         para = torch.Tensor(parameters)
#                         upper_tri_list_M = Model_fine_tuning.TD_layer_transfer(para, P_last_phase, P, indices)
#                         model.Son_swapnet.fc3.diagonal = nn.Parameter(upper_tri_list_M)
#             print("Phase " + str(phase-1) + " opponent model has been transferred to the one in phase " + str(phase))
#         return model
#
#     @staticmethod
#     def TD_layer_transfer(upper_tri_list_N, N, M, index_list):
#         if (N > M):
#             Matrix = np.zeros([N, N])
#             ind = 0
#             for i in range(N):
#                 for j in range(N):
#                     if (j > i):
#                         Matrix[i, j] = upper_tri_list_N[ind]
#                         ind += 1
#
#             Matrix = Matrix[index_list, :]
#             Matrix = Matrix[:, index_list]
#             upper_tri_list_M = []
#             for i in range(M):
#                 for j in range(M):
#                     if (j > i):
#                         upper_tri_list_M.append(Matrix[i, j])
#             return torch.Tensor(upper_tri_list_M)
#
#         elif (N < M):
#             # 计算 N×N 矩阵上三角元素的数量（不包括对角线）
#             num_elements_N = (N * (N - 1)) // 2
#
#             # 如果给定的列表长度与计算出的上三角元素数量不匹配，则抛出异常
#             if len(upper_tri_list_N) != num_elements_N:
#                 raise ValueError("给定的列表长度与 N×N 矩阵上三角元素的数量不匹配")
#
#                 # 初始化 M×M 矩阵上三角元素的列表，全部填充为0
#             num_elements_M = (M * (M - 1)) // 2
#             upper_tri_list_M = [1] * num_elements_M
#
#             # 映射 N×N 矩阵的上三角元素到 M×M 矩阵的上三角元素
#             index_N = 0  # 用于遍历 upper_tri_list_N 的索引
#             index_M_current = 0  # 用于在当前 M×M 上三角元素列表中插入元素的索引
#             for i in range(M):
#                 for j in range(i + 1, M):
#                     # 如果 i 和 j 都在 N×N 矩阵的范围内内，则映射元素
#                     if i < N and j < N:
#                         upper_tri_list_M[index_M_current] = upper_tri_list_N[index_N]
#                         index_N += 1
#                         # 无论如何，我们都要增加 index_M_current 来移动到下一个位置
#                     index_M_current += 1
#
#                     # 此时，如果 N < M，则 upper_tri_list_M 的后半部分已经是0（因为我们初始化为0了）
#             # 如果 N == M，则所有元素都已经被正确映射，无需额外操作
#             return torch.Tensor(upper_tri_list_M)
#
#     @staticmethod
#     def train_multiple_models_mp(train_loader, model, num_epochs, trainable_layers):
#         trainable_layers = [['fc1'], ['fc3'], ['ae1'], ['fc1', 'fc3'], ['fc1', 'ae1'], ['fc3', 'ae1'],
#                             ['fc1', 'fc3', 'ae1'], ['all']]
#
#         parallel_fine_tune_num = len(trainable_layers)
#         model_copies = [copy.deepcopy(model) for _ in range(parallel_fine_tune_num)]
#         args_list = [(train_loader, model_copies[i], num_epochs, trainable_layers[i])
#                      for i in range(parallel_fine_tune_num)]
#
#         with Pool(processes=parallel_fine_tune_num) as pool:
#             results = pool.starmap(NN_model.model_fine_tune, args_list)
#         return zip(*results)
#
#     @staticmethod
#     def similarity_calculation(test_loader, model1, model2):
#         for i in range(len(model1)):
#             model1[i].eval()
#             model2[i].eval()
#
#         total_relative_error = 0.0
#         total_samples = 0
#
#         with torch.no_grad():
#             if(len(model1) == 1):
#                 for inputs1, inputs2, _ in test_loader:
#                     outputs_model1 = model1[0](inputs1, inputs2)
#                     outputs_model2 = model2[0](inputs1, inputs2)
#
#                     bias = outputs_model1 - outputs_model2
#                     eps = 1e-8  # 防止除以0
#                     relative_error = torch.abs(bias) / (torch.abs(outputs_model1) + eps)
#
#                     total_relative_error += relative_error.sum()
#                     total_samples += relative_error.numel()
#             else:
#                 for inputs, _ in test_loader:
#                     inputs = inputs.float()
#
#                     input_explicit = model1[0](inputs)
#                     input_explicit_pow = torch.stack([input_explicit ** i for i in range(model1[1].order)], dim=-1)
#                     output_explicit = model1[1](input_explicit, input_explicit_pow)[:, :, 0]
#                     outputs_model1 = model1[2](output_explicit)
#
#                     input_explicit = model2[0](inputs)
#                     input_explicit_pow = torch.stack([input_explicit ** i for i in range(model2[1].order)], dim=-1)
#                     output_explicit = model2[1](input_explicit, input_explicit_pow)[:, :, 0]
#                     outputs_model2 = model2[2](output_explicit)
#
#                     bias = outputs_model1 - outputs_model2
#                     eps = 1e-8  # 防止除以0
#                     relative_error = torch.abs(bias) / (torch.abs(outputs_model1) + eps)
#
#                     total_relative_error += relative_error.sum()
#                     total_samples += relative_error.numel()
#
#         final_mean_relative_error = total_relative_error / total_samples
#         similarity = 1 / (1 + final_mean_relative_error)
#         return similarity
#
#     @staticmethod
#     def fidelity_calculation(test_loader, model):
#         for net in model:
#             net.eval()
#         criterion = nn.MSELoss()
#         total_loss = 0.0
#         total_samples = 0
#
#         with torch.no_grad():  # 不计算梯度，节省显存和计算资源
#             if len(model) == 1:
#                 for inputs1, inputs2, targets in test_loader:
#                     outputs = model[0](inputs1, inputs2)
#
#                     loss = criterion(outputs.reshape(-1, targets.shape[1]), targets)
#
#                     total_loss += loss.item() * targets.size(0)  # 累加损失
#                     total_samples += targets.size(0)
#             else:
#                 for inputs, targets in train_loader:
#                     inputs = inputs.float()
#                     targets = targets.float()
#
#                     input_explicit = model[0](inputs)
#                     input_explicit_pow = torch.stack([input_explicit ** i for i in range(model[1].order)], dim=-1)
#                     output_explicit = model[1](input_explicit, input_explicit_pow)[:, :, 0]
#                     outputs = model[2](output_explicit)
#
#                     loss = criterion(outputs.reshape(-1, targets.shape[1]), targets)
#
#                     total_loss += loss.item() * targets.size(0)  # 累加损失
#                     total_samples += targets.size(0)
#
#         return total_loss / total_samples  # 返回平均损失，即误差越小保真度越高

learned_decp = []
learned_impo = []
for i in range(len(parameter.P)):
    P = parameter.P[i]
    ae2 = NN_model.DiagonalLinear(channels=P)
    ae2.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae2_Adv_phase=" + str(i) + "_deception.pkl"))
    # print(1/ae1.diagonal)
    learned_dec = (ae2.diagonal).detach().numpy()
    learned_dec = learned_dec / np.sum(learned_dec) * np.sum(parameter.DLi[i])
    learned_decp.append(learned_dec)
    # learned_decp.append((ae2.diagonal).detach().numpy())
    # print((ae2.diagonal).detach().numpy())

    model = NN_model.Opponent_model_explicit(channels=P, order=3)
    model.load_state_dict(torch.load(model_path + str(i) + ".pkl"))  # 执行方对任务方的建模
    # print(1 / model.Son_swapnet.fc1.diagonal)
    learned_impo.append((1 / model.Son_swapnet.fc1.diagonal).detach().numpy())
    print(model.Son_swapnet.fc3.diagonal)
for i in range(len(parameter.P)):
    learned_impo[i] = learned_impo[i] / np.sum(learned_impo[i]) * parameter.P[i]
    print(learned_impo[i])
    print(np.var(learned_impo[i]))

real_impo = parameter.PNi
Adv_time = parameter.Adv_time


for i in range(6):
    real = []
    learn = []
    time = []
    for j in range(len(real_impo)):
        real.append(real_impo[j][i])
        learn.append(learned_impo[j][i])

        real.append(real_impo[j][i])
        learn.append(learned_impo[j][i])

        time.append(Adv_time[j])
        if(j<len(Adv_time)-1):
            time.append(Adv_time[j+1])
        else:
            time.append(40)

    fig, ax = plt.subplots(figsize=(5.5, 1.5))
    plt.plot(time, real, color="#37A4F6", linewidth=4.0)
    plt.plot(time, learn, color="#DB5F57", linewidth=4.0, linestyle='--')

    plt.xlim(0., 35)
    plt.ylim(0.7, 1.3)
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=0)
    plt.yticks([])
    plt.tight_layout()
    plt.show()


for i in range(6, 8):
    real = []
    learn = []
    time = []
    for j in range(len(real_impo)-2):
        real.append(real_impo[j][i])
        learn.append(learned_impo[j][i])

        real.append(real_impo[j][i])
        learn.append(learned_impo[j][i])

        time.append(Adv_time[j])
        if(j<len(Adv_time)-3):
            time.append(Adv_time[j+1])
        else:
            real.append(0)
            learn.append(0)
            time.append(20.0)
            time.append(20.0)

    fig, ax = plt.subplots(figsize=(5.5, 1.5))
    plt.plot(time, real, color="#37A4F6", linewidth=4.0)
    plt.plot(time, learn, color="#DB5F57", linewidth=4.0, linestyle='--')

    plt.xlim(0., 35)
    plt.ylim(0.7, 1.3)
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=0)
    plt.yticks([])
    plt.tight_layout()
    plt.show()

real_decp = parameter.DLi
Adv_time = parameter.Adv_time


for i in range(6):
    real = []
    learn = []
    time = []
    for j in range(len(real_decp)):
        real.append(real_decp[j][i])
        learn.append(learned_decp[j][i])

        real.append(real_decp[j][i])
        learn.append(learned_decp[j][i])

        time.append(Adv_time[j])
        if(j<len(Adv_time)-1):
            time.append(Adv_time[j+1])
        else:
            time.append(40)

    fig, ax = plt.subplots(figsize=(5.5, 1.5))
    plt.plot(time, real, color="#37A4F6", linewidth=4.0)
    plt.plot(time, learn, color="#E6A326", linewidth=4.0, linestyle='--')

    plt.xlim(0., 35)
    plt.ylim(0.7, 1.3)
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=0)
    plt.yticks([])
    plt.tight_layout()
    plt.show()


for i in range(6, 8):
    real = []
    learn = []
    time = []
    for j in range(len(real_decp)-2):
        real.append(real_decp[j][i])
        learn.append(learned_decp[j][i])

        real.append(real_decp[j][i])
        learn.append(learned_decp[j][i])

        time.append(Adv_time[j])
        if(j<len(Adv_time)-3):
            time.append(Adv_time[j+1])
        else:
            real.append(0)
            learn.append(0)
            time.append(20.0)
            time.append(20.0)

    fig, ax = plt.subplots(figsize=(5.5, 1.5))
    plt.plot(time, real, color="#37A4F6", linewidth=4.0)
    plt.plot(time, learn, color="#E6A326", linewidth=4.0, linestyle='--')

    plt.xlim(0., 35)
    plt.ylim(0.7, 1.3)
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=0)
    plt.yticks([])
    plt.tight_layout()
    plt.show()