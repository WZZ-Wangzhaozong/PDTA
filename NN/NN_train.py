import random
import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import NN.NN_model as NN_model
import parameter
from collections import defaultdict
import torch

def mask_matrix_by_labels(matrix, labels):
    P = len(labels)
    assert matrix.shape == (P, P), "matrix must be P x P"

    label_groups = defaultdict(list)
    for idx, label in enumerate(labels):
        label_groups[label].append(idx)

    masked = np.full_like(matrix, fill_value=10000)

    for group_indices in label_groups.values():
        for i in group_indices:
            for j in group_indices:
                masked[i, j] = matrix[i, j]

    return masked

Adv_phase = 2
P = parameter.P[Adv_phase] if Adv_phase >= 0 else parameter.P[0]  # Opponents' number
PNi = parameter.PNi[Adv_phase] if Adv_phase >= 0 else np.array([np.average(parameter.PNi[0]) for _ in range(P)])  # Importance of opponents
Connect_graph = parameter.Connect_graph[Adv_phase] if Adv_phase >= 0 else [0 for i in range(P)]  # Connectivity of opponents
Bij = parameter.Bij[np.ix_(parameter.opponents_index[Adv_phase], parameter.opponents_index[Adv_phase])] \
    if Adv_phase >= 0 else parameter.Bij[np.ix_(parameter.opponents_index[0], parameter.opponents_index[0])]
Bij = mask_matrix_by_labels(Bij, Connect_graph)  # Transfer damping between opponents

KE = parameter.KE if Adv_phase >= 0 else parameter.KE * random.uniform(0.7, 1.3)  # Transfer ability factor
alpha = parameter.exp if Adv_phase >= 0 else parameter.exp * random.uniform(0.7, 1.3)  # Transfer index factor
KR = parameter.KR if Adv_phase >= 0 else parameter.KR * random.uniform(0.7, 1.3)  # Recovery ability factor
beta = parameter.beta if Adv_phase >= 0 else parameter.beta * random.uniform(0.7, 1.3)  # Recovery index factor
order = parameter.order  # Number of elements in beta
num_samples = 1000

model = NN_model.Opponent_model_explicit(channels=P, order=order)  # Opponent model without considering deception

# Model pre-training
# KE = 0.0  # if NN structure is TNNLS-based
args = (KE, KR, PNi, Bij, alpha, beta)

# Generating data used for pre-training
x1, x2, y = NN_model.Data_create_big(P, args, seg=10, len=100, shuffle=True)
loader = DataLoader(TensorDataset(x1, x2, y), batch_size=64, shuffle=False)

if KE == 0.0:  # if NN structure is TNNLS-based
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

    NN_model.model_train(loader, model, num_epochs=1000, optimizer=optimizer)
    torch.save(model.state_dict(), sys.path[0] + r"\trained_NN\tnnls_based_Adv_phase=" + str(Adv_phase) + ".pkl")
else:  # if NN structure is not TNNLS-based
    NN_model.model_train(loader, model, num_epochs=1000)
    torch.save(model.state_dict(), sys.path[0] + r"\trained_NN\Adv_phase=" + str(Adv_phase) + ".pkl")

# Model testing
NN_model.print_parameters_in_model(model)
x1, x2, y = NN_model.Data_create_big(P, args, seg=2, len=100, shuffle=True)
loader = DataLoader(TensorDataset(x1, x2, y), batch_size=64, shuffle=False)
NN_model.model_test(loader, model)