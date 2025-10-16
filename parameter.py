import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import NN.NN_model as NN_model
import sys

'''Hyperparameters'''
Searchwolf_num = 100  # Total number of search wolves
Max_iter = 10  # Maximum number of iterations
fine_tuning_epochs = 2000  # fine-tuning epochs

'''Environment location'''
script_dir = os.path.dirname(os.path.abspath(__file__))
all_sheets = pd.read_excel(script_dir + r"\data_save\excel_file\env.xlsx", sheet_name=None, header=None)
obstacles = all_sheets["obstacles"].to_numpy()
opponents = all_sheets["opponents"].to_numpy()
base = all_sheets["base"].to_numpy()[0]

'''Settings of both sides'''
end_symbol = 0.05  # Symbol of the end of the adversarial process
C = 0.024  # Weakening ability of single agent
Agent_number = 50  # Agents' number
KO = Agent_number * C  # Total weakening ability of the allies' side

# The beginning time of each phase
Duration = 1.0  # Duration allowed for the opponent's adjustment
# For example: from 8.0-8.0+duration, the opponents are connected for adjusting deployment
Adv_time = [0.0, 8.0, 8.0+Duration, 15.0, 15.0+Duration, 20.0, 20.0+Duration]

# Opponents' number in each phase
P = [8, 8, 8, 8, 8, 6, 6]
x0 = torch.ones([P[0]]) * 2.0  # Initial anti-capabilities of opponents
# Opponent index
opponents_index = [[0, 1, 2, 3, 4, 5, 6, 7],
                   [0, 1, 2, 3, 4, 5, 6, 7],
                   [0, 1, 2, 3, 4, 5, 6, 7],
                   [0, 1, 2, 3, 4, 5, 6, 7],
                   [0, 1, 2, 3, 4, 5, 6, 7],
                   [0, 1, 2, 3, 4, 5],
                   [0, 1, 2, 3, 4, 5]]
KE = 0.16  # Transfer ability factor
exp = 3.0  # Transfer index factor
KR = 0.012  # Recovery ability factor
beta = np.array([1.0, 0.6, 1.1])  # Recovery index factor
order = 3  # Number of elements in beta

all_sheets = pd.read_excel(script_dir + r"\data_save\excel_file\flight_point.xlsx", sheet_name=None, header=None)["path_len"].to_numpy()
# Travelling time
Time = (all_sheets[0, :] / max(all_sheets[0, :P[0]])).tolist()
# Transfer damping in each phase
Bij = np.vstack([all_sheets[1:, :], np.zeros([max(P)])])
Bij = (Bij + Bij.T + np.diag([1 for i in range(max(P))])) / max(all_sheets[0, :P[0]])

# Connectivity in each phase
Connect_graph = [[0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 1, 1, 1],
                 [0, 1, 0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 0, 1, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1],
                 [0, 0, 0, 1, 1, 1]]

# Importance in each phase
PNi = [np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
       np.array([0.9, 0.9, 1.1, 1.1, 0.9, 0.9, 1.1, 1.1]),
       np.array([0.9, 0.9, 1.1, 1.1, 0.9, 0.9, 1.1, 1.1]),
       np.array([0.9, 0.9, 1.0, 1.2, 0.9, 0.9, 1.0, 1.2]),
       np.array([0.9, 0.9, 1.0, 1.2, 0.9, 0.9, 1.0, 1.2]),
       np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
       np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]

# Deception in each phase
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

        torch.save(ae1.state_dict(), sys.path[0] + r"\NN\trained_NN\ae1_Adv_phase=" + str(phase) + ".pkl")
        torch.save(ae2.state_dict(), sys.path[0] + r"\NN\trained_NN\ae2_Adv_phase=" + str(phase) + ".pkl")