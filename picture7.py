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

strategies = ["prediction-driven", "prediction-driven-NN_deception"]
sheet_names = ["act_seq", "allies_s1", "opponent_exc", "opponent_rec", "opponent_exc_vel", "opponent_rec_vel", "opponents"]
color = ['#E6A326', '#6EDB57', '#37A4F6', "#DB5F57"]

fig, ax = plt.subplots(figsize=(7.5*5.18/2.24, 7.5))

file = sys.path[0] + "\\data\\" + "prediction-driven" + ".xlsx"
data = pd.read_excel(file, header=None, sheet_name='opponents')
data = np.array(data)
for i in range(data.shape[0]-1, 1):
    data[i, 1:] = (data[i, 1:] - data[i - 1, 1:]) / (data[i, 0] - data[i - 1, 0])

file = sys.path[0] + "\\data\\" + "prediction-driven-NN_deception" + ".xlsx"
data_NN = pd.read_excel(file, header=None, sheet_name='opponents')
data_NN = np.array(data_NN)
for i in range(data_NN.shape[0]-1, 1):
    data_NN[i, 1:] = (data_NN[i, 1:] - data_NN[i - 1, 1:]) / (data_NN[i, 0] - data_NN[i - 1, 0])


for i in range(8):
    plt.subplot(8, 1, i+1)
    plt.plot(data[:, 0], data[:, i+1], color="#37A4F6", linewidth=8.0)
    plt.plot(data_NN[:, 0], data_NN[:, i+1], color="#DB5F57", linewidth=8.0, linestyle='--')

    plt.xlim(0.0, 48.0)
    plt.ylim(0.2, 2.3)
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=0)
    plt.yticks([])
    # plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=0))
    plt.tight_layout()
plt.show()



