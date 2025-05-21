import random
from scipy.integrate import solve_ivp, odeint
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import NN.NN_model as NN_model
import parameter
import dill as pickle
from collections import defaultdict
import torch

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

Adv_phase = 2
P = parameter.P[Adv_phase] if Adv_phase >= 0 else parameter.P[0]  # 任务点个数
PNi = parameter.PNi[Adv_phase] if Adv_phase >= 0 else np.array([np.average(parameter.PNi[0]) for _ in range(P)])  # 任务点重要度
Connect_graph = parameter.Connect_graph[Adv_phase] if Adv_phase >= 0 else [0 for i in range(P)]  # Updating connectivity of opponents
Bij = parameter.Bij[np.ix_(parameter.opponents_index[Adv_phase], parameter.opponents_index[Adv_phase])] \
    if Adv_phase >= 0 else parameter.Bij[np.ix_(parameter.opponents_index[0], parameter.opponents_index[0])]
Bij = mask_matrix_by_labels(Bij, Connect_graph)  # Updating exchange difficulty of opponents

KE = parameter.KE if Adv_phase >= 0 else parameter.KE * random.uniform(0.8, 1.2)  # 任务方交换能力
alpha = parameter.exp if Adv_phase >= 0 else parameter.exp * random.uniform(0.8, 1.2)  # 任务方交换因子
KR = parameter.KR if Adv_phase >= 0 else parameter.KR * random.uniform(0.8, 1.2)  # 任务方恢复能力
beta = parameter.beta if Adv_phase >= 0 else parameter.beta * random.uniform(0.8, 1.2)  # 任务方恢复因子
num_samples = 1000
order = parameter.order

model = NN_model.Opponent_model_explicit(channels=P, order=order)

# 模型训练
# KE = 0.0
args = (KE, KR, PNi, Bij, alpha, beta)

x1, x2, y = NN_model.Data_create_big(P, args, seg=10, len=100, shuffle=True)
loader = DataLoader(TensorDataset(x1, x2, y), batch_size=64, shuffle=False)

if KE == 0.0:
    frozen_layers = ['fc1', 'fc2', 'fc3']
    trainable_params = []
    for name, param in model.named_parameters():
        if any(ln in name for ln in frozen_layers):
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable_params.append(param)
    model.Son_swapnet.fc1.diagonal = nn.Parameter(torch.zeros(P))
    optimizer = optim.Adam(trainable_params, lr=0.001)

    NN_model.model_train(loader, model, 1000, optimizer)
    torch.save(model.state_dict(), sys.path[0] + r"\trained_NN\tnnls_based_Adv_phase=" + str(Adv_phase) + ".pkl")
else:
    NN_model.model_train(loader, model, 1000)
    torch.save(model.state_dict(), sys.path[0] + r"\trained_NN\Adv_phase=" + str(Adv_phase) + ".pkl")

NN_model.print_parameters_in_model(model)

# 模型测试
x1, x2, y = NN_model.Data_create_big(P, args, seg=2, len=100, shuffle=True)
loader = DataLoader(TensorDataset(x1, x2, y), batch_size=64, shuffle=False)
NN_model.model_test(loader, model)