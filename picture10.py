
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


# TNNLS = [49.843, 49.015, 48.849, 44.183, 1000]
# TASE = [35.699, 35.788, 35.926, 46.062, 1000]
# our = [30.449, 29.949, 29.499, 38.199, 35.160]
# our_NN = [30.999, 30.113, 29.978, 39.302, 36.168]
TNNLS = [48.849, 49.843, 49.015, 44.183]
TASE = [35.926, 35.699, 35.788,  46.062, 52.167]
our = [29.499, 30.449, 29.949, 38.199, 39.302]
our_NN = [29.978, 30.999, 30.113,  39.302, 40.415]

print(np.average(TNNLS))
print(np.average(TASE))
print(np.average(our))
print(np.average(our_NN))
TNNLS = [48.849, 49.843, 49.015, 44.183, 1000]
TASE = [35.926, 35.699, 35.788,  46.062, 52.167]
our = [29.499, 30.449, 29.949, 38.199, 39.302]
our_NN = [29.978, 30.999, 30.113,  39.302, 40.415]


"#E6A326""#6EDB57""#37A4F6""#37A4F6"
fig, ax = plt.subplots(figsize=(7.5*2.32, 7.5))
width = 1.0
for i in range(5):
    ax.bar(7 * i + width * 0.0, TNNLS[i], width=width, color="#E6A326")
    ax.bar(7 * i + width * 1.2, TASE[i], width=width, color="#6EDB57")
    ax.bar(7 * i + width * 2.4, our[i], width=width, color="#37A4F6")
    ax.bar(7 * i + width * 3.6, our_NN[i], width=width, color="#37A4F6", hatch='/')
ax.set_ylim(0, 80)
ax.set_xticks([])
# ax.set_yticks([])
# plt.ylim(0, 550)
plt.yticks(fontsize=0.0)
plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
plt.tight_layout()
plt.show()