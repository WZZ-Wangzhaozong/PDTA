import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import NN.NN_model as NN_model
import sys

'''GWO Hyperparameters'''
Searchwolf_num = 100  # 狼群规模
Max_iter = 10  # 搜索次数

'''Environment location'''
script_dir = os.path.dirname(os.path.abspath(__file__))
all_sheets = pd.read_excel(script_dir + r"\excel_file\env.xlsx", sheet_name=None, header=None)
obstacles = all_sheets["obstacles"].to_numpy()
opponents = all_sheets["opponents"].to_numpy()
base = all_sheets["base"].to_numpy()[0]

'''Settings of both sides'''
x0 = torch.Tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])  # 任务方初始资源量
end_symbol = 0.05  # 对抗结束标志
C = 0.024  # 单个智能体的执行能力
Agent_number = 50  # 执行方智能体数量
KO = Agent_number * C  # 执行方执行能力
KE = 0.16  # 任务方交换能力
exp = 3.0  # 任务方交换因子
KR = 0.012  # 任务方恢复能力
beta = np.array([1.0, 0.6, 1.1])  # 任务方恢复因子
order = 3  # 阶次

P = [8, 8, 8, 8, 8, 6, 6]
opponents_index = [[0, 1, 2, 3, 4, 5, 6, 7],
                   [0, 1, 2, 3, 4, 5, 6, 7],
                   [0, 1, 2, 3, 4, 5, 6, 7],
                   [0, 1, 2, 3, 4, 5, 6, 7],
                   [0, 1, 2, 3, 4, 5, 6, 7],
                   [0, 1, 2, 3, 4, 5],
                   [0, 1, 2, 3, 4, 5]]
Adv_time = [0.0, 8.0, 9.0, 15.0, 16.0, 20.0, 21.0]  # 预训练/实际过程/终止时刻

all_sheets = pd.read_excel(script_dir + r"\excel_file\flight_point.xlsx", sheet_name=None, header=None)["path_len"].to_numpy()

# 任务点额定旅行时长
Time = (all_sheets[0, :] / max(all_sheets[0, :P[0]])).tolist()

# 任务点交换阻尼
Bij = np.vstack([all_sheets[1:, :], np.zeros([max(P)])])
Bij = (Bij + Bij.T + np.diag([1 for i in range(max(P))])) / max(all_sheets[0, :P[0]])
Connect_graph = [[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 1, 1, 1],
                 [0, 1, 0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]]
# print(Time)
# print(Bij)

# 任务点重要度
PNi = [np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
       np.array([0.9, 0.9, 1.1, 1.1, 0.9, 0.9, 1.1, 1.1]),
       np.array([0.9, 0.9, 1.1, 1.1, 0.9, 0.9, 1.1, 1.1]),
       np.array([0.9, 0.9, 1.0, 1.2, 0.9, 0.9, 1.0, 1.2]),
       np.array([0.9, 0.9, 1.0, 1.2, 0.9, 0.9, 1.0, 1.2]),
       np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
       np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]

# 信息虚假程度
DLi = [np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
       np.array([1.1, 1.1, 1.0, 1.0, 1.1, 1.1, 1.0, 1.0]),
       np.array([1.1, 1.1, 1.0, 1.0, 1.1, 1.1, 1.0, 1.0]),
       np.array([1.2, 1.2, 1.0, 1.0, 1.2, 1.2, 1.0, 1.0]),
       np.array([1.2, 1.2, 1.0, 1.0, 1.2, 1.2, 1.0, 1.0]),
       np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
       np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]

if __name__ == '__main__':
    for phase in range(len(DLi)):
        DL = DLi[phase]
        ae1 = NN_model.DiagonalLinear(DL.shape[0])
        ae1.diagonal = nn.Parameter(torch.Tensor(1 / DL))

        ae2 = NN_model.DiagonalLinear(DL.shape[0])
        ae2.diagonal = nn.Parameter(torch.Tensor(DL))

        torch.save(ae1.state_dict(), sys.path[0] + r"\trained_NN\ae1_Adv_phase=" + str(phase) + ".pkl")
        torch.save(ae2.state_dict(), sys.path[0] + r"\trained_NN\ae2_Adv_phase=" + str(phase) + ".pkl")