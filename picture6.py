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

strategies = ["TNNLS", "TASE", "prediction-driven", "prediction-driven-NN_deception", "Quantity-based", "AST", "prediction-driven-NN"]
sheet_names = ["act_seq", "allies_s1", "opponent_exc", "opponent_rec", "opponent_exc_vel", "opponent_rec_vel", "opponents"]
color = ['#E6A326', '#6EDB57', '#37A4F6', "#DB5F57"]

fig, ax = plt.subplots(figsize=(7.5*5.18/2.24, 7.5))
for i in range(4):
    file = sys.path[0] + "\\data\\" + strategies[i] + ".xlsx"
    data = pd.read_excel(file, header=None, sheet_name='opponents')
    data = np.array(data)

    masked = np.where(data[:, 1:] > 0, data[:, 1:], np.inf)
    if i == 3:
        plt.plot(data[:, 0], np.min(masked, axis=1), color=color[i], linewidth=7.0, linestyle='--')
    else:
        plt.plot(data[:, 0], np.min(masked, axis=1), color=color[i], linewidth=7.0)


plt.xlim(0, 35)
plt.ylim(0.05, 2.8)
plt.xticks(fontsize=0)
plt.yticks(fontsize=0)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
plt.grid()
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7.5*5.18/2.24, 7.5))
for i in range(4):
    file = sys.path[0] + "\\data\\" + strategies[i] + ".xlsx"
    data = pd.read_excel(file, header=None, sheet_name='opponents')
    data = np.array(data)
    if i == 3:
        plt.plot(data[:, 0], np.max(data[:, 1:], axis=1), color=color[i], linewidth=7.0, linestyle='--')
    else:
        plt.plot(data[:, 0], np.max(data[:, 1:], axis=1), color=color[i], linewidth=7.0)


plt.xlim(0, 55)
plt.ylim(0.05, 2.8)
plt.xticks(fontsize=0)
plt.yticks(fontsize=0)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
plt.grid()
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(7.5*5.18/2.24, 7.5))
for i in range(4):
    file = sys.path[0] + "\\data\\" + strategies[i] + ".xlsx"
    data = pd.read_excel(file, header=None, sheet_name='opponents')
    data = np.array(data)
    show_data = data[:, 1:]
    non_zero_counts = np.count_nonzero(data[:, 1:], axis=1)
    show_data = show_data.sum(axis=1)
    result = show_data / non_zero_counts

    if i == 3:
        plt.plot(data[:, 0], result, color=color[i], linewidth=7.0, linestyle='--')
    else:
        plt.plot(data[:, 0], result, color=color[i], linewidth=7.0)


plt.xlim(0, 55)
plt.ylim(0.05, 2.8)
plt.xticks(fontsize=0)
plt.yticks(fontsize=0)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
plt.grid()
plt.tight_layout()
plt.show()