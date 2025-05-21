import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import random
from sympy import symbols, Eq, solve
import concurrent.futures
import time as T
import subprocess
import json
import NN.NN_model as NN_model
import Env as Env
import GWO as GWO
import os
import parameter
from collections import defaultdict
from multiprocessing import Pool
import copy
import tkinter as tk
from tkinter import messagebox
import sys

class Data_save:
    @staticmethod
    def Sampling(opponent_his_data, given_intervals=0.05):
        sample_time = 0
        index_list = [0]
        begin = opponent_his_data[0, 0]
        end = opponent_his_data[-1, 0]
        while (True):
            if ((begin + given_intervals * (sample_time + 1)) > end):
                index_list.append(-1)
                break
            else:
                index = np.where(opponent_his_data[:, 0] <= (begin + given_intervals * (sample_time + 1)))[0][-1]
                index_list.append(index)
                sample_time += 1
        return opponent_his_data[index_list, :]

    @staticmethod
    def Calculate_velocity(P, opponent_his_data, KE, KR, PNi, Bij, exp, beta):
        opponent_exc_vel_data = np.empty([0, 1 + P ** 2])
        opponent_rec_vel_data = np.empty([0, 1 + P])
        for i in range(opponent_his_data.shape[0]):
            dxdt1, dxdt2 = Env.backward_equations(opponent_his_data[i, :], KE, KR, PNi, Bij, exp, beta)
            opponent_exc_vel_data = np.vstack([opponent_exc_vel_data, dxdt1])
            opponent_rec_vel_data = np.vstack([opponent_rec_vel_data, dxdt2])
        return opponent_exc_vel_data, opponent_rec_vel_data

    @staticmethod
    def Calculate_amount(opponent_exc_vel_data, opponent_rec_vel_data):
        opponent_exc_data = opponent_exc_vel_data.copy()
        opponent_rec_data = opponent_rec_vel_data.copy()
        for i in range(1, opponent_exc_data.shape[0]):
            opponent_exc_data[i, 1:] = \
                opponent_exc_data[i, 1:] * (
                        opponent_exc_data[i, 0] - opponent_exc_data[i - 1, 0]) + opponent_exc_data[
                                                                                 i - 1, 1:]
        for i in range(1, opponent_exc_data.shape[0]):
            opponent_rec_data[i, 1:] = \
                opponent_rec_data[i, 1:] * (
                        opponent_rec_data[i, 0] - opponent_rec_data[i - 1, 0]) + opponent_rec_data[
                                                                                 i - 1, 1:]
        return opponent_exc_data, opponent_rec_data

    @staticmethod
    def Save_data_as_npy(save_path, file_name, opponent_his_data, opponent_exc_data, opponent_rec_data,
                         opponent_exc_vel_data, opponent_rec_vel_data, action_sequence, allies_s1):
        Adv_phase, KE, KR, KO = file_name
        np.save(
            save_path + "opponents_Adv_phase=" + str(Adv_phase - 1) + "_KE=" + str(KE) + "_KR=" + str(KR)
            + "_KO=" + str(KO) + ".npy", opponent_his_data)

        np.save(
            save_path + "opponent_exc_Adv_phase=" + str(Adv_phase - 1) + "_KE=" + str(KE) + "_KR=" + str(
                KR)
            + "_KO=" + str(KO) + ".npy", opponent_exc_data)

        np.save(
            save_path + "opponent_rec_Adv_phase=" + str(Adv_phase - 1) + "_KE=" + str(KE) + "_KR=" + str(
                KR)
            + "_KO=" + str(KO) + ".npy", opponent_rec_data)

        np.save(save_path + "opponent_exc_vel_Adv_phase=" + str(Adv_phase - 1) + "_KE=" + str(
            KE) + "_KR=" + str(KR)
                + "_KO=" + str(KO) + ".npy", opponent_exc_vel_data)

        np.save(save_path + "opponent_rec_vel_Adv_phase=" + str(Adv_phase - 1) + "_KE=" + str(
            KE) + "_KR=" + str(KR)
                + "_KO=" + str(KO) + ".npy", opponent_rec_vel_data)

        np.save(save_path + "act_seq_Adv_phase=" + str(Adv_phase - 1) + "_KE=" + str(KE) + "_KR=" + str(KR)
                + "_KO=" + str(KO) + ".npy", action_sequence)

        np.save(save_path + "allies_s1_Adv_phase=" + str(Adv_phase - 1) + "_KE=" + str(KE) + "_KR=" + str(KR)
                + "_KO=" + str(KO) + ".npy", allies_s1)

class Initialize_each_phase:
    @staticmethod
    def mask_matrix_by_labels(matrix, labels):
        P = len(labels)
        assert matrix.shape == (P, P), "matrix must be P x P"

        # 构建标签到索引的映射
        label_groups = defaultdict(list)
        for idx, label in enumerate(labels):
            label_groups[label].append(idx)

        # 创建全 10000 的矩阵
        masked = np.full_like(matrix, fill_value=10000)

        # 对每个组，保留组内元素之间的值
        for group_indices in label_groups.values():
            for i in group_indices:
                for j in group_indices:
                    masked[i, j] = matrix[i, j]

        return masked

    @staticmethod
    def Windows_len_select(P, Time):
        if (P <= 4):
            return 6.0 + 48.2755144334742 / 62.08641773198754
        elif (P <= 6):
            return 3.5 + 30.157475345595586 / 62.08641773198754
        else:
            return 3.0 + 23.665621261737517 / 38.304951684997060

    @staticmethod
    def loss_type_select(PNi, Bij, tol_PNi=0.1, tol_Bij=10000):
        PNi = 1 / PNi
        PNi = PNi / np.sum(PNi) * PNi.shape[0]

        if(np.std(PNi) > tol_PNi):
            return "Maximize Decrease"
        elif np.any(Bij) > tol_Bij:
            return "Maximize Decrease"
        return "Minimize Maximum"

class Model_fine_tuning:
    @staticmethod
    def print_parameters_in_model(model):
        print("Parameters in IE network:")
        for name, param in model.Son_swapnet.fc1.named_parameters():
            print(param.data)

        print("Parameters in BC network:")
        for name, param in model.Son_swapnet.fc2.named_parameters():
            print(param.data)

        print("Parameters in TD network:")
        for name, param in model.Son_swapnet.fc3.named_parameters():
            print(param.data)

        print("Parameters in sub-recovery network:")
        for name, param in model.Son_recovernet.fc.named_parameters():
            print(param.data)
        print(" ")

    # @staticmethod
    # def load_ae(retrain, P, P_last_phase, phase):
    #     if(retrain):
    #         if(phase == 0):
    #             ae1 = NN_model.DiagonalLinear(channels=P)
    #             ae2 = NN_model.DiagonalLinear(channels=P)
    #             ae1.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae1_Adv_phase=" + str(phase) + ".pkl"))
    #             ae2.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae2_Adv_phase=" + str(phase) + ".pkl"))
    #         else:
    #             ae1_old = NN_model.DiagonalLinear(channels=P_last_phase)
    #             ae2_old = NN_model.DiagonalLinear(channels=P_last_phase)
    #             ae1_old.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae1_Adv_phase=" + str(phase-1) + ".pkl"))
    #             ae2_old.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae2_Adv_phase=" + str(phase-1) + ".pkl"))
    #             ae1 = NN_model.DiagonalLinear(channels=P)
    #             ae2 = NN_model.DiagonalLinear(channels=P)
    #
    #             if P >= P_last_phase:
    #                 print(phase)
    #                 ae1.diagonal[:P_last_phase] = ae1_old.diagonal
    #                 ae2.diagonal[:P_last_phase] = ae2_old.diagonal
    #             else:
    #                 preserve_index = [parameter.opponents_index[phase-1].index(x) for x in
    #                                   parameter.opponents_index[phase]]
    #                 ae1.diagonal = ae1_old.diagonal[preserve_index]
    #                 ae2.diagonal = ae2_old.diagonal[preserve_index]
    #     else:
    #         ae1 = NN_model.DiagonalLinear(channels=P)
    #         ae2 = NN_model.DiagonalLinear(channels=P)
    #         ae1.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae1_Adv_phase=" + str(phase) + ".pkl"))
    #         ae2.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae2_Adv_phase=" + str(phase) + ".pkl"))
    #     return ae1, ae2

    @staticmethod
    def load_ae(P, phase):
        ae1 = NN_model.DiagonalLinear(channels=P)
        ae2 = NN_model.DiagonalLinear(channels=P)
        ae1.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae1_Adv_phase=" + str(phase) + ".pkl"))
        ae2.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae2_Adv_phase=" + str(phase) + ".pkl"))
        return ae1, ae2

    @staticmethod
    def old_transfer_new_ae(retrain, P, P_last_phase, phase):
        if (retrain):
            if (phase == 0):
                ae1, ae2 = Model_fine_tuning.load_ae(P, phase)
            else:
                ae1_old, ae2_old = Model_fine_tuning.load_ae(P_last_phase, phase-1)
                ae1 = NN_model.DiagonalLinear(channels=P)
                ae2 = NN_model.DiagonalLinear(channels=P)

                if P >= P_last_phase:
                    # with torch.no_grad():
                    #     ae1.diagonal[:P_last_phase] = ae1_old.diagonal
                    #     ae2.diagonal[:P_last_phase] = ae2_old.diagonal
                    # 保留旧模型参数，扩展新的
                    new_diag1 = torch.ones(P, device=ae1_old.diagonal.device)
                    new_diag2 = torch.ones(P, device=ae2_old.diagonal.device)
                    new_diag1[:P_last_phase] = ae1_old.diagonal
                    new_diag2[:P_last_phase] = ae2_old.diagonal
                    ae1.diagonal = nn.Parameter(new_diag1)
                    ae2.diagonal = nn.Parameter(new_diag2)
                else:
                    preserve_index = [parameter.opponents_index[phase - 1].index(x) for x in
                                      parameter.opponents_index[phase]]
                    # with torch.no_grad():
                    #     ae1.diagonal = ae1_old.diagonal[preserve_index]
                    #     ae2.diagonal = ae2_old.diagonal[preserve_index]
                    new_diag1 = ae1_old.diagonal[preserve_index]
                    new_diag2 = ae2_old.diagonal[preserve_index]
                    ae1.diagonal = nn.Parameter(new_diag1)
                    ae2.diagonal = nn.Parameter(new_diag2)
        else:
            ae1, ae2 = Model_fine_tuning.load_ae(P, phase)
        return ae1, ae2

    @staticmethod
    def load_model(P, order, phase):
        model = NN_model.Opponent_model_explicit(channels=P, order=order)
        model.load_state_dict(torch.load(model_path + str(phase) + ".pkl"))  # 执行方对任务方的建模
        return model

    @staticmethod
    def old_transfer_new(retrain, P, P_last_phase, Adv_phase):
        opponents_number_change = P - P_last_phase
        if (not retrain):
            print("Phase " + str(Adv_phase) + " opponent model being used")
            model = Model_fine_tuning.load_model(P, order, Adv_phase)
            Model_fine_tuning.print_parameters_in_model(model)
        else:
            print("Phase " + str(Adv_phase-1) + " opponent model being transferred")
            if (opponents_number_change == 0):
                model = Model_fine_tuning.load_model(P, order, Adv_phase-1)
            else:
                model_old = Model_fine_tuning.load_model(P_last_phase, order, Adv_phase-1)
                model = NN_model.Opponent_model_explicit(channels=P, order=order)

                # Transfer parameters in old network to new network
                with torch.no_grad():
                    # Recovery network transfer
                    model.Son_recovernet.fc.load_state_dict(model_old.Son_recovernet.fc.state_dict())

                    # IE network transfer
                    indices = [parameter.opponents_index[Adv_phase-1].index(x) for x in parameter.opponents_index[Adv_phase]]
                    if (opponents_number_change < 0):
                        model.Son_swapnet.fc1.diagonal.data.copy_(model_old.Son_swapnet.fc1.diagonal.data[indices])
                    else:
                        model.Son_swapnet.fc1.diagonal.data[:P_last_phase].copy_(model_old.Son_swapnet.fc1.diagonal.data)
                        model.Son_swapnet.fc1.diagonal.data[P_last_phase:] = model_old.Son_swapnet.fc1.diagonal.data.sum().item() / P_last_phase

                    # BC network transfer
                    model.Son_swapnet.fc2.load_state_dict(model_old.Son_swapnet.fc2.state_dict())

                    # TD network transfer
                    for parameters in model_old.Son_swapnet.fc3.parameters():  # 打印出参数矩阵及值
                        para = torch.Tensor(parameters)
                        upper_tri_list_M = Model_fine_tuning.TD_layer_transfer(para, P_last_phase, P, indices)
                        model.Son_swapnet.fc3.diagonal = nn.Parameter(upper_tri_list_M)
            print("Phase " + str(Adv_phase-1) + " opponent model has been transferred to the one in phase " + str(Adv_phase) )
        return model

    @staticmethod
    def TD_layer_transfer(upper_tri_list_N, N, M, index_list):
        if (N > M):
            Matrix = np.zeros([N, N])
            ind = 0
            for i in range(N):
                for j in range(N):
                    if (j > i):
                        Matrix[i, j] = upper_tri_list_N[ind]
                        ind += 1

            Matrix = Matrix[index_list, :]
            Matrix = Matrix[:, index_list]
            upper_tri_list_M = []
            for i in range(M):
                for j in range(M):
                    if (j > i):
                        upper_tri_list_M.append(Matrix[i, j])
            return torch.Tensor(upper_tri_list_M)

        elif (N < M):
            # 计算 N×N 矩阵上三角元素的数量（不包括对角线）
            num_elements_N = (N * (N - 1)) // 2

            # 如果给定的列表长度与计算出的上三角元素数量不匹配，则抛出异常
            if len(upper_tri_list_N) != num_elements_N:
                raise ValueError("给定的列表长度与 N×N 矩阵上三角元素的数量不匹配")

                # 初始化 M×M 矩阵上三角元素的列表，全部填充为0
            num_elements_M = (M * (M - 1)) // 2
            upper_tri_list_M = [1] * num_elements_M

            # 映射 N×N 矩阵的上三角元素到 M×M 矩阵的上三角元素
            index_N = 0  # 用于遍历 upper_tri_list_N 的索引
            index_M_current = 0  # 用于在当前 M×M 上三角元素列表中插入元素的索引
            for i in range(M):
                for j in range(i + 1, M):
                    # 如果 i 和 j 都在 N×N 矩阵的范围内内，则映射元素
                    if i < N and j < N:
                        upper_tri_list_M[index_M_current] = upper_tri_list_N[index_N]
                        index_N += 1
                        # 无论如何，我们都要增加 index_M_current 来移动到下一个位置
                    index_M_current += 1

                    # 此时，如果 N < M，则 upper_tri_list_M 的后半部分已经是0（因为我们初始化为0了）
            # 如果 N == M，则所有元素都已经被正确映射，无需额外操作
            return torch.Tensor(upper_tri_list_M)

    @staticmethod
    def train_multiple_models_mp(train_loader, model, num_epochs, trainable_layers):
        trainable_layers = [['fc1'], ['fc3'], ['ae1'], ['fc1', 'fc3'], ['fc1', 'ae1'], ['fc3', 'ae1'],
                            ['fc1', 'fc3', 'ae1'], ['all']]

        parallel_fine_tune_num = len(trainable_layers)
        model_copies = [copy.deepcopy(model) for _ in range(parallel_fine_tune_num)]
        args_list = [(train_loader, model_copies[i], num_epochs, trainable_layers[i])
                     for i in range(parallel_fine_tune_num)]

        with Pool(processes=parallel_fine_tune_num) as pool:
            results = pool.starmap(NN_model.model_fine_tune, args_list)
        return zip(*results)

    @staticmethod
    def similarity_calculation(test_loader, model1, model2):
        for i in range(len(model1)):
            model1[i].eval()
            model2[i].eval()

        total_relative_error = 0.0
        total_samples = 0

        with torch.no_grad():
            if(len(model1) == 1):
                for inputs1, inputs2, _ in test_loader:
                    outputs_model1 = model1[0](inputs1, inputs2)
                    outputs_model2 = model2[0](inputs1, inputs2)

                    bias = outputs_model1 - outputs_model2
                    eps = 1e-8  # 防止除以0
                    relative_error = torch.abs(bias) / (torch.abs(outputs_model1) + eps)

                    total_relative_error += relative_error.sum()
                    total_samples += relative_error.numel()
            else:
                for inputs, _ in test_loader:
                    inputs = inputs.float()

                    input_explicit = model1[0](inputs)
                    input_explicit_pow = torch.stack([input_explicit ** i for i in range(model1[1].order)], dim=-1)
                    output_explicit = model1[1](input_explicit, input_explicit_pow)[:, :, 0]
                    outputs_model1 = model1[2](output_explicit)

                    input_explicit = model2[0](inputs)
                    input_explicit_pow = torch.stack([input_explicit ** i for i in range(model2[1].order)], dim=-1)
                    output_explicit = model2[1](input_explicit, input_explicit_pow)[:, :, 0]
                    outputs_model2 = model2[2](output_explicit)

                    bias = outputs_model1 - outputs_model2
                    eps = 1e-8  # 防止除以0
                    relative_error = torch.abs(bias) / (torch.abs(outputs_model1) + eps)

                    total_relative_error += relative_error.sum()
                    total_samples += relative_error.numel()

        final_mean_relative_error = total_relative_error / total_samples
        similarity = 1 / (1 + final_mean_relative_error)
        return similarity

    @staticmethod
    def fidelity_calculation(test_loader, model):
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():  # 不计算梯度，节省显存和计算资源
            for inputs1, inputs2, targets in test_loader:
                outputs, _, _ = model(inputs1, inputs2)
                loss = criterion(outputs.reshape(-1, targets.shape[1]), targets)
                total_loss += loss.item() * inputs1.size(0)  # 累加损失
                total_samples += inputs1.size(0)

        return total_loss / total_samples  # 返回平均损失，即误差越小保真度越高

def main(strategy, deception=False):
    if strategy not in ["Quantity-based", "AST", "TNNLS", "TASE", "prediction-driven", "prediction-driven-NN", "prediction-driven-NN_deception"]:
        print("Please check whether the strategy name is correct...")
        print("请检查策略名称是否正确...")
        sys.exit()

    data_path = data_path_base + "\\" + strategy + "\\"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    '''Adversarial process initialization'''
    allies_s1 = np.zeros([4, 20000])  # Determined actions [opponent index, return moment, arrival moment, Agent number/Total number]
    allies_s2 = np.zeros([2, 20000])  # Undetermined actions [depart moment, Agent number/Total number]
    allies_s2[:, 0] = np.array([0.0, 1.0])
    opponents_s3 = x0  # Anti-capability sequence of opponents

    s1_len = 0  # Length of valid information in allies_s1
    s2_len = 1  # Length of valid information in allies_s2

    start = 0.0
    Adv_phase = 0  # Adversarial phase
    train_time = 0.0

    retrain = False
    deception = False
    if strategy in ["prediction-driven-NN", "prediction-driven-NN_deception"]:
        retrain = True
    if strategy in ["prediction-driven-NN_deception"]:
        deception = True

    while (torch.any(opponents_s3 > end_symbol)):  # Sign of the end of the adversarial process
        if(Adv_phase < len(Adv_time) and start >= Adv_time[Adv_phase]):  # The opponent strategy changes and the adversarial process enters a new phase
            print("进入第" + str(Adv_phase) + "对抗阶段...")
            print("Entering confrontation phase " + str(Adv_phase) + "...")
            print(" ")
            if(Adv_phase >= 1):
                # Sample the adversarial process data at intervals of 0.05 seconds.
                opponent_his_data = Data_save.Sampling(opponent_his_data, given_intervals=0.05)
                # Calculate the exchange and recovery velocity of opponents.
                opponent_exc_vel_data, opponent_rec_vel_data = Data_save.Calculate_velocity(P, opponent_his_data, KE, KR, PNi, Bij, exp, beta)
                # Calculate the exchange and recovery amount of opponents.
                opponent_exc_data, opponent_rec_data = Data_save.Calculate_amount(opponent_exc_vel_data, opponent_rec_vel_data)
                # Save data as npy.
                Data_save.Save_data_as_npy(data_path, [Adv_phase, KE, KR, KO], opponent_his_data, opponent_exc_data, opponent_rec_data,
                                 opponent_exc_vel_data, opponent_rec_vel_data, action_sequence, allies_s1)

            P_last_phase = parameter.P[Adv_phase-1] if Adv_phase > 0 else parameter.P[0]  # Number of opponents in last phase
            P = parameter.P[Adv_phase]  # Updating the number of opponents
            opponents_number_change = P - P_last_phase  # Change in number of opponents
            opponents_index = parameter.opponents_index[Adv_phase]  # Opponent index
            Time = np.array(parameter.Time)[opponents_index]  # Updating travel time from base to opponents
            PNi = parameter.PNi[Adv_phase]  # Updating the importance of opponents
            DLi = parameter.DLi[Adv_phase]  # Updating the deception level of opponents

            Connect_graph = parameter.Connect_graph[Adv_phase]  # Updating connectivity of opponents
            Bij = parameter.Bij[np.ix_(parameter.opponents_index[Adv_phase], parameter.opponents_index[Adv_phase])]
            Bij = Initialize_each_phase.mask_matrix_by_labels(Bij, Connect_graph)  # Updating exchange difficulty of opponents

            args = (KE, KR, PNi, Bij, exp, beta, end_symbol)

            win_len = Initialize_each_phase.Windows_len_select(P, Time)
            task = Env.task_symbol(P, Time, win_len)  # Instantiate a sequence of deduction window tasks
            loss_type = Initialize_each_phase.loss_type_select(PNi, Bij)
            if(Adv_phase==0):
                loss_type = "Minimize Maximum"
            else:
                loss_type = "Maximize Decrease"

            # Generate a new round of data containers
            opponent_his_data = np.empty([0, P+1])  # [time, {opponents}]
            action_sequence = np.empty([0, P+1])  # [time, {opponents}]

            if(opponents_number_change > 0):
                x0_new = torch.full((opponents_number_change, ), 2.0)
                opponents_s3 = torch.cat((opponents_s3, x0_new), dim=0)
            elif(opponents_number_change < 0):
                preserve_index = [parameter.opponents_index[Adv_phase-1].index(x) for x in opponents_index]
                abandon_index = [i for i, x in enumerate(parameter.opponents_index[Adv_phase-1]) if x not in opponents_index]
                preserved_values = opponents_s3[preserve_index]
                adjustment = opponents_s3[abandon_index].sum() / preserved_values.shape[0]
                opponents_s3 = preserved_values + adjustment

            start = Adv_time[Adv_phase]
            end = allies_s2[0, 0].item()
            opponents_s3, opponent_his_data = Env.Env_recursion(start, end, np.empty([4, 0]), KO,
                                                                opponents_s3, opponent_his_data, args)

            if strategy == "TNNLS":
                # Select the model structure of TNNLS
                model = NN_model.Opponent_model_explicit(channels=P, order=order)
                model.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\tnnls_based_Adv_phase=" + str(Adv_phase) + ".pkl"))
                Model_fine_tuning.print_parameters_in_model(model)
                opponent_model = [model]
            else:
                retrain = True if strategy in ["prediction-driven-NN", "prediction-driven-NN_deception"] else False

                model = Model_fine_tuning.old_transfer_new(retrain, P, P_last_phase, Adv_phase)  # Pre-trained model/previous stage model
                ae1, ae2 = Model_fine_tuning.old_transfer_new_ae(retrain, P, P_last_phase, Adv_phase)  # Pre-trained AE/previous stage AE
                if deception:
                    opponent_model = [ae1, model, ae2]
                else:
                    opponent_model = [model]

                target_model = Model_fine_tuning.load_model(P, order, Adv_phase)  # Well-trained model
                target_ae1, target_ae2 = Model_fine_tuning.load_ae(P, Adv_phase)  # Well-trained AE
                if deception:
                    target_opponent_model = [target_ae1, target_model, target_ae2]
                else:
                    target_opponent_model = [target_model]

            Adv_phase += 1  # Update phase number
            retrain_number = 0

        # Sorting the order of identified tasks according to the moment of arrival at the adversary
        sort = np.argsort(allies_s1[2, :s1_len])
        allies_s1[:, :s1_len] = allies_s1[:, sort]

        # Prioritization of undetermined tasks according to the moment of return to base
        sort = np.argsort(allies_s2[0, :s2_len])
        allies_s2[:, :s2_len] = allies_s2[:, sort]

        merged_matrix = Env.merge_columns_with_tolerance(allies_s2[:, :s2_len])
        s2_len = merged_matrix.shape[1]
        allies_s2[:, :s2_len] = merged_matrix

        # Extract the tasks that have been identified but not executed and time-align them
        start = allies_s2[0, 0].item()
        cols = np.where(allies_s1[2, :s1_len] > start)[0]
        allies_s1_current_round = allies_s1[:, cols.tolist()].copy()
        allies_s1_current_round[2, :] -= start

        KO_action = KO * allies_s2[1, 0].item()  # allies_s2[1, 0]为当前能调用的智能体数量/总数
        print("实际能力：" + str(KO_action))
        Agent_tobe_departed_number = int(round(allies_s2[1, 0].item() * Agent_number))  # 本批次智能体数量
        print("本批次智能体数量:" + str(Agent_tobe_departed_number))

        s1 = T.time()
        if deception:
            opponent_observed = opponents_s3 * DLi
        else:
            opponent_observed = opponents_s3

        if (strategy == "Quantity-based"):
            print("优化目标:Quantity-based")
            u_opt = (opponent_observed / sum(opponent_observed)).detach().numpy()
        elif (strategy == "AST"):
            print("优化目标:AST")
            result = [y / x for x, y in zip(Time, opponent_observed)]
            u_opt = result / np.sum(result)
        elif (strategy == "TASE"):
            print("优化目标:Minimize Maximum")
            u_opt, score = GWO.DEGWO(P, Searchwolf_num, Max_iter, KO_action, opponent_observed, Time, end_symbol,
                                     torch.Tensor(allies_s1_current_round), task, opponent_model, win_len,
                                     loss_type="Minimize Maximum")
            u_opt = u_opt.detach().numpy()
        elif (strategy == "TNNLS"):
            print("优化目标:Minimize Maximum")
            u_opt, score = GWO.DEGWO(P, Searchwolf_num, Max_iter, KO_action, opponent_observed, Time, end_symbol,
                                     torch.Tensor(allies_s1_current_round), task, opponent_model, win_len,
                                     loss_type="Minimize Maximum")
            u_opt = u_opt.detach().numpy()
        elif (strategy == "prediction-driven" or strategy == "prediction-driven-NN"):
            print("优化目标:" + str(loss_type))
            u_opt, score = GWO.DEGWO(P, Searchwolf_num, Max_iter, KO_action, opponent_observed, Time, end_symbol,
                                     torch.Tensor(allies_s1_current_round), task, opponent_model, win_len, loss_type)
            u_opt = u_opt.detach().numpy()
        print("优化在现实世界中的耗时:" + str(T.time() - s1))

        '''连续分配量转离散数量值'''
        decimal_array = u_opt * Agent_tobe_departed_number  # 智能体数量数组：小数
        integer_array = np.floor(decimal_array).astype("int")  # 智能体数量数组：整数
        res = max(0, Agent_tobe_departed_number - np.sum(integer_array).item())  # 剩余待分配的智能体数量

        while (res != 0):
            gaps = decimal_array - integer_array  # 差距
            indices = np.argsort(gaps)  # 从小到大排序的索引
            integer_array[indices[-1]] += 1
            res -= 1

        print("最优分配：" + str(integer_array))
        action_sequence = np.vstack([action_sequence, np.concatenate([[start], integer_array])])

        '''更新allies_s1和allies_s2'''
        # 具体执行智能体编号confirmed_index和unconfirmed_index
        Eff_actions_index = np.where(integer_array != 0)[0]
        Eff_actions_len = len(Eff_actions_index)

        array1 = np.array([[i + 1, start + Time[i] * 2, start + Time[i],
                            integer_array[i] / Agent_number] for i in Eff_actions_index]).T
        allies_s1[:, s1_len:(s1_len + Eff_actions_len)] = array1

        array2 = np.array([[start + Time[i] * 2,
                            integer_array[i] / Agent_number] for i in Eff_actions_index]).T
        allies_s2[:, s2_len:(s2_len + Eff_actions_len)] = array2

        s1_len += Eff_actions_len
        s2_len += Eff_actions_len

        # 删除上一轮次未确定但在本轮次已执行的任务
        s2_len -= 1
        allies_s2 = allies_s2[:, 1:]

        # 将环境递归到下一轮次开始时刻
        end = np.min(allies_s2[0, :s2_len]).item()
        if Adv_phase < len(Adv_time):
            end = min(end, Adv_time[Adv_phase])
        cols = np.where((allies_s1[2, :] > start) & (allies_s1[2, :] <= end))[0]
        action_sequence_current_round = allies_s1[:, cols].copy()

        print("打击前：" + str(opponents_s3))
        print("仿真世界起始时刻：" + str(start))
        print("仿真世界结束时刻：" + str(end))
        opponents_s3, opponent_his_data = Env.Env_recursion(start, end, action_sequence_current_round,
                                                            KO, opponents_s3, opponent_his_data, args)
        print("打击后：" + str(opponents_s3))
        print(" ")
        start = end

        # 模型重训练
        if (retrain and ((opponent_his_data[-1, 0] - train_time) >= 0.2)):
            train_time = opponent_his_data[-1, 0].copy()

            retrain_data = opponent_his_data.copy()
            retrain_label = torch.Tensor(retrain_data[:, 1:])
            for e in range(retrain_data.shape[0]):
                retrain_label[e, :] = Env.forward_equations(None, retrain_data[e, 1:], KE, KR, PNi, Bij, exp, beta, None)

            if deception:
                retrain_data = torch.Tensor(retrain_data[:, 1:] * DLi)
                retrain_label = retrain_label * DLi
                train_loader = DataLoader(TensorDataset(retrain_data, retrain_label), batch_size=64, shuffle=True)
                trainable_layers = [['fc1'], ['fc3'], ['ae1'], ['fc1', 'fc3'], ['fc1', 'ae1'], ['fc3', 'ae1'],
                                    ['fc1', 'fc3', 'ae1'], ['all']]
            else:
                retrain_data1 = torch.Tensor(retrain_data[:, 1:])
                retrain_data2 = torch.zeros([retrain_data.shape[0], P, 3])
                for o in range(3):
                    retrain_data2[:, :, o] = retrain_data1 ** o
                retrain_label = retrain_label
                train_loader = DataLoader(TensorDataset(retrain_data1, retrain_data2, retrain_label), batch_size=64,
                                          shuffle=True)
                trainable_layers = [['fc1'], ['fc3'], ['fc1', 'fc3'], ['all']]

            parallel_fine_tune_num = len(trainable_layers)
            model_copies = [copy.deepcopy(opponent_model) for _ in range(parallel_fine_tune_num)]
            args_list = [(train_loader, model_copies[i], 2000, trainable_layers[i]) for i in
                         range(parallel_fine_tune_num)]

            with Pool(processes=parallel_fine_tune_num) as pool:
                results = pool.starmap(NN_model.model_fine_tune, args_list)
            model_sequence, loss_sequence = list(zip(*results))

            fidelities = [Model_fine_tuning.fidelity_calculation(train_loader, m) for m in model_sequence]
            max_fidelity_index = min(range(len(fidelities)), key=lambda i: fidelities[i])
            opponent_model, loss = model_sequence[max_fidelity_index], loss_sequence[max_fidelity_index]

            if not deception:
                torch.save(opponent_model[0].state_dict(),
                           model_path + str(Adv_phase-1) + "_retrain_number=" + str(retrain_number) + ".pkl")
            else:
                torch.save(opponent_model[0].state_dict(),
                           sys.path[0] + r"\trained_NN\ae1_Adv_phase=" + str(Adv_phase-1) + ".pkl")
                torch.save(opponent_model[1].state_dict(),
                           model_path + str(Adv_phase-1) + "_retrain_number=" + str(retrain_number) + ".pkl")
                torch.save(opponent_model[2].state_dict(),
                           sys.path[0] + r"\trained_NN\ae2_Adv_phase=" + str(Adv_phase-1) + ".pkl")

            np.save(data_path + "loss_Adv_phase=" + str(Adv_phase-1) + "_retrain_number=" + str(retrain_number)
                    + "_KE=" + str(KE) + "_KR=" + str(KR) + "_KO=" + str(KO) + ".npy", np.array(loss))

            # 网络停止训练
            similarity = Model_fine_tuning.similarity_calculation(train_loader, target_opponent_model, opponent_model)
            if similarity > 0.95 or opponent_his_data[-1, 0] > (opponent_his_data[0, 0] + 1.0):
                retrain = False
                model = Model_fine_tuning.load_model(P, order, Adv_phase-1)
                ae1, ae2 = Model_fine_tuning.load_ae(P, Adv_phase-1)
                opponent_model = [model]
                if deception:
                    opponent_model = [ae1, model, ae2]

            retrain_number += 1

    # Sample the adversarial process data at intervals of 0.05 seconds.
    opponent_his_data = Data_save.Sampling(opponent_his_data, given_intervals=0.05)
    # Calculate the exchange and recovery velocity of opponents.
    opponent_exc_vel_data, opponent_rec_vel_data = Data_save.Calculate_velocity(P, opponent_his_data, KE, KR, PNi, Bij, exp, beta)
    # Calculate the exchange and recovery amount of opponents.
    opponent_exc_data, opponent_rec_data = Data_save.Calculate_amount(opponent_exc_vel_data, opponent_rec_vel_data)
    # Save data as npy.
    Data_save.Save_data_as_npy(data_path, [Adv_phase, KE, KR, KO], opponent_his_data, opponent_exc_data, opponent_rec_data,
                           opponent_exc_vel_data, opponent_rec_vel_data, action_sequence, allies_s1)

'''GWO Hyperparameters'''
Searchwolf_num = parameter.Searchwolf_num  # 狼群规模
Max_iter = parameter.Max_iter  # 搜索次数

'''Confrontation Start, Process and End Settings'''
x0 = parameter.x0  # 任务方初始资源量
end_symbol = parameter.end_symbol  # 对抗结束标志
Adv_time = parameter.Adv_time

'''Allies and Opponents Capability Parameters'''
KO = parameter.KO  # 执行方执行能力
Agent_number = parameter.Agent_number  # 执行方智能体数量
KE = parameter.KE  # 任务方交换能力
exp = parameter.exp  # 任务方交换因子
KR = parameter.KR  # 任务方恢复能力
beta = parameter.beta  # 任务方恢复因子
order = parameter.order

model_path = sys.path[0] + r"\trained_NN\Adv_phase="
data_path_base = sys.path[0] + r"\data"