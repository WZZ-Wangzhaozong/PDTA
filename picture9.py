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
from collections import defaultdict
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

retrain = [1, 1, 5, 1, 5, 1, 1]
# for i in range(len(parameter.P)):
#     for j in range(retrain[i]):
#         model_path = sys.path[0] + r"\trained_NN\Adv_phase=" + str(i) + "_retrain_number=" + str(j) + "_deception.pkl"
#         model = NN_model.Opponent_model_explicit(channels=parameter.P[i], order=3)
#         model.load_state_dict(torch.load(model_path))
#
#         P = parameter.P[i]
#         PNi = parameter.PNi[i]
#         Connect_graph = parameter.Connect_graph[i]
#         Bij = parameter.Bij[np.ix_(parameter.opponents_index[i], parameter.opponents_index[i])]
#         Bij = mask_matrix_by_labels(Bij, Connect_graph)  # Updating exchange difficulty of opponents
#
#         KE = parameter.KE
#         alpha = parameter.exp
#         KR = parameter.KR
#         beta = parameter.beta
#         num_samples = 1000
#         order = parameter.order
#         args = (KE, KR, PNi, Bij, alpha, beta)
#
#         x1, x2, y = NN_model.Data_create_big(parameter.P[i], args, seg=5, len=100, shuffle=True)
#         loader = DataLoader(TensorDataset(x1, x2, y), batch_size=64, shuffle=False)
#         NN_model.model_test(loader, model)

loss = [0.1641, 0.014,
        0.0, 0.0,
        0.1241, 0.0314, 0.0384, 0.0223, 0.0085, 0.0021,
        0.0, 0.0,
        0.0514, 0.0012, 0.0043, 0.0023, 0.0032, 0.0024,
        0.0, 0.0,
        0.0711, 0.0014]
Time = parameter.Adv_time

Adv_time = [0.0, 1.306,
            1.306, 8.01,
            8.01, 8.11, 8.2, 8.27, 8.47, 8.53,
            8.53, 15.06,
            15.06, 15.14, 15.24, 15.33, 15.45, 15.68,
            15.68, 20.04,
            20.04, 20.23]

T1 = [0.0, 0.25, 0.25]
L1 = [0.1641, 0.014, 0.0]

T2 = [0.25, 8.01,]
L2 = [ 0.0, 0.0, ]

T3 = [8.01, 8.01, 8.11, 8.2, 8.27, 8.47, 8.53, 8.53]
L3 = [0.0, 0.1241, 0.0314, 0.0384, 0.0223, 0.0085, 0.0021, 0.0]

T4 = [ 8.53, 15.06,]
L4 = [0.0, 0.0, ]

T5 = [15.06, 15.06, 15.14, 15.24, 15.33, 15.45, 15.68, 15.68]
L5 = [0.0, 0.0514, 0.0012, 0.0043, 0.0023, 0.0032, 0.0024, 0.0]

T6 = [15.68, 20.04, ]
L6 = [0.0, 0.0, ]

T7 = [20.04, 20.04, 20.23, 20.23]
L7 = [0.0, 0.0711, 0.0014, 0.0]

T8 = [20.23, 50]
L8 = [0.0, 0.0]

LINEWI = 8.0
fig, ax = plt.subplots(figsize=(7.5*5.18/2.24, 7.5))
plt.plot(T1, L1, color="#DB5F57", linewidth=LINEWI)
plt.plot(T2, L2, color="#37A4F6", linewidth=LINEWI)
plt.plot(T3, L3, color="#DB5F57", linewidth=LINEWI)
plt.plot(T4, L4, color="#37A4F6", linewidth=LINEWI)
plt.plot(T5, L5, color="#DB5F57", linewidth=LINEWI)
plt.plot(T6, L6, color="#37A4F6", linewidth=LINEWI)
plt.plot(T7, L7, color="#DB5F57", linewidth=LINEWI)
plt.plot(T8, L8, color="#37A4F6", linewidth=LINEWI)
plt.xlim(0.0, 32.0)
plt.ylim(-0.015, 0.18)
plt.xticks(fontsize=0)
plt.yticks(fontsize=0)
plt.yticks([])
plt.tight_layout()
plt.grid()
plt.show()
