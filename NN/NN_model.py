import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Grandson_net(nn.Module):
    def __init__(self, order):
        super(Grandson_net, self).__init__()
        self.order = order
        self.fc = nn.Sequential(
            nn.Linear(self.order, 1, bias=False),
            nn.LeakyReLU())

    def forward(self, x):
        return self.fc(x)

class Son_recovernet(nn.Module):
    def __init__(self, channels, order):
        super(Son_recovernet, self).__init__()
        self.channels = channels
        self.order = order
        self.fc = Grandson_net(self.order)

    def forward(self, x):  # x.shape=[64, channels, order]
        outputs = torch.zeros((x.shape[0], self.channels, 1))
        for i in range(self.channels):
            net = self.fc
            outputs[:, i, :] = net(x[:, i, :])
        return outputs

class DiagonalLinear(nn.Module):
    def __init__(self, channels, bias=False, inverse=False, source_layer=None):
        super(DiagonalLinear, self).__init__()
        self.inverse = inverse
        self.source_layer = source_layer

        if not inverse:
            self.diagonal = nn.Parameter(torch.ones(channels))
        else:
            self.diagonal = None
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        if self.inverse:
            diagonal = 1.0 / self.source_layer.diagonal
        else:
            diagonal = self.diagonal

        diag_mat = torch.diag_embed(diagonal)
        out = torch.matmul(x, diag_mat)
        return out

class Son_swapnet(nn.Module):
    def __init__(self, channels, order, mat, diff_indices):
        super(Son_swapnet, self).__init__()
        self.channels = channels
        self.order = order
        self.mat = mat
        self.dim = int(self.channels*(self.channels-1)/2)
        self.diff_indices = diff_indices
        self.fc1 = DiagonalLinear(self.channels)
        self.fc2 = Grandson_net(self.order)
        self.fc3 = DiagonalLinear(self.dim)

    def forward(self, x):
        # Importance evaluation
        outputs1 = self.fc1(x)  # output δ

        # Bias calculation
        diffs = torch.matmul(outputs1, self.diff_indices)  # output δi-δj
        Sign = torch.sign(diffs).reshape(-1, self.dim, 1)  # output sgn(δi-δj)
        Abs = torch.abs(diffs)  # output abs(δi-δj)
        outputs2 = torch.stack([Abs ** (i + 1) for i in range(self.order)], dim=2)  # output [abs(δi-δj), (δi-δj)^2, abs(δi-δj)^3]

        outputs3 = torch.zeros([x.shape[0], self.dim, 1])
        for i in range(self.dim):
            outputs3[:, i, :] = self.fc2(outputs2[:, i, :])  # output alpha^[abs(δi-δj)]-1
        outputs3 = torch.mul(outputs3, Sign)  # output sgn(δi-δj)*alpha^[abs(δi-δj)]-1
        outputs3 = self.fc3(outputs3.reshape(-1, self.dim)).reshape(-1, self.dim, 1)  # output {sgn(δi-δj)*alpha^[abs(δi-δj)]-1}/Bij

        # Summing related component in output3
        outputs4 = torch.zeros([x.shape[0], self.channels, 1])
        for i in range(self.channels):
            outputs4[:, i, :] -= torch.sum(outputs3[:, self.mat[i][0], :], dim=1)
            outputs4[:, i, :] += torch.sum(outputs3[:, self.mat[i][1], :], dim=1)
        return outputs4

# class Net(nn.Module):
#     def __init__(self, channels, order):
#         super(Net, self).__init__()
#         self.channels = channels
#         self.order = order
#         self.mat = self.matrix(channels)
#         self.diff_indices = self.matrix1(channels)
#
#         self.Son_swapnet = Son_swapnet(self.channels, self.order, self.mat, self.diff_indices)
#         self.Son_recovernet = Son_recovernet(self.channels, self.order)
#
#     def forward(self, x1, x2):  # x.shape=[64, channels, 1]
#         outputs = self.Son_swapnet(x1) + self.Son_recovernet(x2)
#         return outputs
#
#     def forward_special(self, x1, x2):  # x.shape=[64, channels, 1]
#         outputs = self.Son_swapnet(x1) + self.Son_recovernet(x2)
#         return outputs, self.Son_swapnet(x1), self.Son_recovernet(x2)
#
#     def matrix(self, channels):
#         mat = []
#         for i in range(-1, channels - 1):
#             mat.append([])
#             row = i
#             col = i + 1
#             index1 = np.arange(row + 1, channels - 1, 1)
#             if (index1.shape[0] != 0):
#                 index1 = index1 - min(index1)
#             S = 0
#             for index3 in range(col):
#                 S += (channels - 1 - index3)
#             jian = (S + index1).astype(int)
#
#             mat[i + 1].append(jian)
#             jia = []
#             a = row
#             for index3 in range(row + 1):
#                 jia.append(a)
#                 S = channels - 2 - index3
#                 a = a + S
#             jia = np.array(jia).astype(int)
#             mat[i + 1].append(jia)
#         return mat
#
#     def matrix1(self, channels):
#         P = channels
#         diff_index = torch.triu_indices(P, P, offset=1)
#         diff_indices = torch.zeros([P, int(P * (P - 1) / 2)])
#         for i in range(int(P * (P - 1) / 2)):
#             diff_indices[diff_index[0, i], i] = 1.0
#             diff_indices[diff_index[1, i], i] = -1.0
#         return diff_indices

class Opponent_model_explicit(nn.Module):
    def __init__(self, channels, order):
        super(Opponent_model_explicit, self).__init__()
        self.channels = channels
        self.order = order
        self.mat = self.matrix(channels)
        self.diff_indices = self.matrix1(channels)

        self.Son_swapnet = Son_swapnet(self.channels, self.order, self.mat, self.diff_indices)
        self.Son_recovernet = Son_recovernet(self.channels, self.order)

    def forward(self, x1, x2):  # x.shape=[64, channels, 1]
        outputs = self.Son_swapnet(x1) + self.Son_recovernet(x2)
        return outputs

    def forward_special(self, x1, x2):  # x.shape=[64, channels, 1]
        outputs = self.Son_swapnet(x1) + self.Son_recovernet(x2)
        return outputs, self.Son_swapnet(x1), self.Son_recovernet(x2)

    def matrix(self, channels):
        mat = []
        for i in range(-1, channels - 1):
            mat.append([])
            row = i
            col = i + 1
            index1 = np.arange(row + 1, channels - 1, 1)
            if (index1.shape[0] != 0):
                index1 = index1 - min(index1)
            S = 0
            for index3 in range(col):
                S += (channels - 1 - index3)
            jian = (S + index1).astype(int)

            mat[i + 1].append(jian)
            jia = []
            a = row
            for index3 in range(row + 1):
                jia.append(a)
                S = channels - 2 - index3
                a = a + S
            jia = np.array(jia).astype(int)
            mat[i + 1].append(jia)
        return mat

    def matrix1(self, channels):
        P = channels
        diff_index = torch.triu_indices(P, P, offset=1)
        diff_indices = torch.zeros([P, int(P * (P - 1) / 2)])
        for i in range(int(P * (P - 1) / 2)):
            diff_indices[diff_index[0, i], i] = 1.0
            diff_indices[diff_index[1, i], i] = -1.0
        return diff_indices

def system_equations(t, x, KE, KR, PNi, Bij, exp, beta):
    dim = x.shape[0]

    for i in range(dim):
        x[i] = max(0, x[i])

    # Swapping mechanism
    dxdt1 = np.zeros([dim])
    if (KE != 0.0):
        for i in range(dim):
            for j in range(dim):
                dxdt1[i] += (np.sign(x[j] / PNi[j] - x[i] / PNi[i]) * (exp ** abs(x[j] / PNi[j] - x[i] / PNi[i]) - 1) / \
                             Bij[i, j])
    dxdt1 *= KE

    # Recovery mechanism
    dxdt2 = (beta[0] + x * beta[1] + x ** 2 * beta[2]) * KR
    return dxdt1 + dxdt2

def Data_create_big(P, args, seg, len, shuffle, order=3):
    if (shuffle == True):
        print("Genearting data for pre-training...")
    else:
        print("Genearting data for testing...")

    Process_train_data1 = np.empty([0, P])
    Process_train_data2 = np.empty([0, P, order])
    Process_train_label = np.empty([0, P])
    for i in range(seg):
        x0 = np.random.rand(P)
        x0 = (x0 - min(x0)) / (max(x0) - min(x0)) * 0.5 + 0.1

        for index in range(1, int(10E4)):
            t_span = [0, index]
            t_eval = np.linspace(t_span[0], t_span[1], 5000)

            sol = solve_ivp(system_equations, t_span, x0, t_eval=t_eval, args=args)
            # If the opponents without the recovery ability
            if (args[1] > 0.0 and (sol.y[:, -1] >= 2.0).any()):
                time_index = 10E4
                for p in range(P):
                    if(np.where(sol.y[p, :] >= 1.0)[0].shape[0]!=0):
                        time_index = min(time_index, np.where(sol.y[p, :] >= 1.0)[0][0] - 1)
                t_span = [0, sol.t[time_index]]

                t_eval = np.linspace(t_span[0], t_span[1], len)
                sol = solve_ivp(system_equations, t_span, x0, t_eval=t_eval, args=args)
                break
            # If the opponents possessing the recovery ability
            elif (args[1] == 0.0 and (sol.y[:, -1].var() <= 10E-5)):
                t_span = [0, sol.t[-1]]

                t_eval = np.linspace(t_span[0], t_span[1], len)  # 评估时间点
                sol = solve_ivp(system_equations, t_span, x0, t_eval=t_eval, args=args)
                break

        dydx = sol.y.copy()
        for index in range(dydx.shape[1]):
            dydx[:, index] = system_equations(None, sol.y[:, index], args[0], args[1], args[2], args[3], args[4], args[5])

        Process_train_data1 = np.vstack([Process_train_data1, sol.y.T])

        add_matrix = np.zeros([len, P, order])
        for index in range(order):
            add_matrix[:, :, index] = sol.y.T ** (index)
        Process_train_data2 = np.vstack([Process_train_data2, add_matrix])

        Process_train_label = np.vstack([Process_train_label, dydx.T])
    return torch.Tensor(Process_train_data1), torch.Tensor(Process_train_data2), torch.Tensor(Process_train_label)

def Data_create_small(P, args, seg, len, shuffle):
    if (shuffle == True):
        print("Genearting data for pre-training...")
    else:
        print("Genearting data for testing...")

    Process_train_data1 = np.empty([0, P])
    Process_train_data2 = np.empty([0, P, args[6]])
    Process_train_label = np.empty([0, P])
    for i in range(seg):
        x0 = np.random.rand(P)
        x0 = (x0 - min(x0)) / (max(x0) - min(x0)) * 0.05 + 1.98  # *1.95+0.005

        for index in range(1, int(10E4)):
            t_span = [0, index]
            t_eval = np.linspace(t_span[0], t_span[1], 4000)  # 评估时间点

            sol = solve_ivp(system_equations, t_span, x0, t_eval=t_eval, args=args)

            if (args[1] > 0.0 and (sol.y[:, -1] >= 2.0).any()):
                print(sol.y.shape)
                time_index = 10E4
                for p in range(P):
                    if(np.where(sol.y[p, :] >= 0.1)[0].shape[0]!=0):
                        time_index = min(time_index, np.where(sol.y[p, :] >= 0.1)[0][0] - 1)
                t_span = [0, sol.t[time_index]]

                t_eval = np.linspace(t_span[0], t_span[1], len)
                sol = solve_ivp(system_equations, t_span, x0, t_eval=t_eval, args=args)
                break

            elif (args[1] == 0.0 and (sol.y[:, -1].var() <= 10E-5)):
                t_span = [0, sol.t[-1]]

                t_eval = np.linspace(t_span[0], t_span[1], len)  # 评估时间点
                sol = solve_ivp(system_equations, t_span, x0, t_eval=t_eval, args=args)
                break

        dydx = sol.y.copy()
        for index in range(dydx.shape[1]):
            dydx[:, index] = system_equations(None, sol.y[:, index], args[0], args[1], args[2], args[3], args[4], args[5], args[6])

        Process_train_data1 = np.vstack([Process_train_data1, sol.y.T])

        add_matrix = np.zeros([len, P, args[6]])
        for index in range(args[6]):
            add_matrix[:, :, index] = sol.y.T ** (index)
        Process_train_data2 = np.vstack([Process_train_data2, add_matrix])

        Process_train_label = np.vstack([Process_train_label, dydx.T])
    return torch.Tensor(Process_train_data1), torch.Tensor(Process_train_data2), torch.Tensor(Process_train_label)

def model_train(train_loader, model, num_epochs, optimizer=None):
    print("Model training...")

    criterion = nn.MSELoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    Loss = []
    for epoch in range(num_epochs):
        L1 = 0.0
        for inputs1, inputs2, targets in train_loader:
            outputs = model(inputs1, inputs2)
            loss = criterion(outputs.reshape(-1, targets.shape[1]), targets)
            L1 += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            Loss.append(L1)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')
    return sum(Loss) / (num_epochs / 10)

def model_fine_tune(train_loader, model, num_epochs, trainable_layers=None):
    print("Fine-tuning " + str(trainable_layers) + "...")

    trainable_params = []
    if(len(model) == 1):
        if trainable_layers == ['all']:
            for name, param in model[0].named_parameters():
                param.requires_grad = True
                trainable_params.append(param)
        else:
            for name, param in model[0].named_parameters():
                if any(ln in name for ln in trainable_layers):
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
    else:
        if trainable_layers in [['fc1'], ['fc3'], ['fc1', 'fc3']]:
            for name, param in model[1].named_parameters():
                if any(ln in name for ln in trainable_layers):
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
        elif trainable_layers in [['ae1'], ['fc1', 'ae1'], ['fc3', 'ae1'], ['fc1', 'fc3', 'ae1']]:
            for name, param in model[0].named_parameters():
                param.requires_grad = True
                trainable_params.append(param)

            for name, param in model[1].named_parameters():
                if any(ln in name for ln in trainable_layers):
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
        elif trainable_layers == ['all']:
            for name, param in model[0].named_parameters():
                param.requires_grad = True
                trainable_params.append(param)

            for name, param in model[1].named_parameters():
                param.requires_grad = True
                trainable_params.append(param)

    optimizer = optim.Adam(trainable_params, lr=0.001)
    criterion = nn.MSELoss()

    Loss = []
    for epoch in range(num_epochs):
        L1 = 0.0
        if len(model) > 1:
            for inputs, targets in train_loader:
                inputs = inputs
                targets = targets
                input_explicit = model[0](inputs)
                input_explicit_pow = torch.stack([input_explicit ** i for i in range(model[1].order)], dim=-1)
                output_explicit = model[1](input_explicit, input_explicit_pow)[:, :, 0]
                with torch.no_grad():
                    model[2].diagonal.copy_(1.0 / model[0].diagonal)
                outputs = model[2](output_explicit)

                loss = criterion(outputs.reshape(-1, targets.shape[1]), targets)
                L1 += float(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            for inputs1, inputs2, targets in train_loader:
                inputs1 = inputs1.float()
                inputs2 = inputs2.float()
                targets = targets.float()

                outputs = model[0](inputs1, inputs2)
                loss = criterion(outputs.reshape(-1, targets.shape[1]), targets)
                L1 += float(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if (epoch + 1) % 500 == 0:
            Loss.append(L1)
    return model, Loss

def model_test(test_loader, model):
    print("Model testing...")
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    len = 0

    plt.figure()
    with torch.no_grad():
        for inputs1, inputs2, targets in test_loader:
            outputs = model(inputs1, inputs2)

            loss = criterion(outputs.reshape(-1, targets.shape[1]), targets)
            for i in range(targets.shape[1]):
                plt.subplot(targets.shape[1], 1, 1 + i)
                plt.plot(np.linspace(len, len + targets.shape[0], targets.shape[0]),
                         targets[:, i], c='r', linewidth=2.1, label='real')
                plt.plot(np.linspace(len, len + targets.shape[0], targets.shape[0]),
                         outputs[:, i], c='b', linewidth=2, label='pred')
            len += targets.shape[0]
            total_loss += loss.item() * inputs1.size(0)
    plt.legend()
    plt.show()

    rmses = []
    with torch.no_grad():
        for inputs1, inputs2, targets in test_loader:
            outputs = model(inputs1, inputs2)
            rmse = torch.sqrt(torch.mean((outputs.reshape(-1, targets.shape[1]) - targets) ** 2, dim=1))
            rmses.extend(rmse.tolist())
    avg_rmse = np.mean(rmses)
    print(f'Test RMSE: {avg_rmse:.4f}')

# def TD_layer_transfer(upper_tri_list_N, N, M, index_list):
#     if(N > M):
#         Matrix = np.zeros([N, N])
#         ind = 0
#         for i in range(N):
#             for j in range(N):
#                 if (j > i):
#                     Matrix[i, j] = upper_tri_list_N[ind]
#                     ind += 1
#
#         Matrix = Matrix[index_list, :]
#         Matrix = Matrix[:, index_list]
#         upper_tri_list_M = []
#         for i in range(M):
#             for j in range(M):
#                 if (j > i):
#                     upper_tri_list_M.append(Matrix[i, j])
#         return torch.Tensor(upper_tri_list_M)
#
#     elif(N < M):
#         # 计算 N×N 矩阵上三角元素的数量（不包括对角线）
#         num_elements_N = (N * (N - 1)) // 2
#
#         # 如果给定的列表长度与计算出的上三角元素数量不匹配，则抛出异常
#         if len(upper_tri_list_N) != num_elements_N:
#             raise ValueError("给定的列表长度与 N×N 矩阵上三角元素的数量不匹配")
#
#             # 初始化 M×M 矩阵上三角元素的列表，全部填充为0
#         num_elements_M = (M * (M - 1)) // 2
#         upper_tri_list_M = [1] * num_elements_M
#
#         # 映射 N×N 矩阵的上三角元素到 M×M 矩阵的上三角元素
#         index_N = 0  # 用于遍历 upper_tri_list_N 的索引
#         index_M_current = 0  # 用于在当前 M×M 上三角元素列表中插入元素的索引
#         for i in range(M):
#             for j in range(i + 1, M):
#                 # 如果 i 和 j 都在 N×N 矩阵的范围内内，则映射元素
#                 if i < N and j < N:
#                     upper_tri_list_M[index_M_current] = upper_tri_list_N[index_N]
#                     index_N += 1
#                     # 无论如何，我们都要增加 index_M_current 来移动到下一个位置
#                 index_M_current += 1
#
#                 # 此时，如果 N < M，则 upper_tri_list_M 的后半部分已经是0（因为我们初始化为0了）
#         # 如果 N == M，则所有元素都已经被正确映射，无需额外操作
#         return torch.Tensor(upper_tri_list_M)

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

# def map_upper_triangular_to_larger_matrix(upper_tri_list_N, N, M):
#     # 计算 N×N 矩阵上三角元素的数量（不包括对角线）
#     num_elements_N = (N * (N - 1)) // 2
#
#     # 如果给定的列表长度与计算出的上三角元素数量不匹配，则抛出异常
#     if len(upper_tri_list_N) != num_elements_N:
#         raise ValueError("给定的列表长度与 N×N 矩阵上三角元素的数量不匹配")
#
#         # 初始化 M×M 矩阵上三角元素的列表，全部填充为0
#     num_elements_M = (M * (M - 1)) // 2
#     upper_tri_list_M = [1] * num_elements_M
#
#     # 映射 N×N 矩阵的上三角元素到 M×M 矩阵的上三角元素
#     index_N = 0  # 用于遍历 upper_tri_list_N 的索引
#     index_M_current = 0  # 用于在当前 M×M 上三角元素列表中插入元素的索引
#     for i in range(M):
#         for j in range(i + 1, M):
#             # 如果 i 和 j 都在 N×N 矩阵的范围内内，则映射元素
#             if i < N and j < N:
#                 upper_tri_list_M[index_M_current] = upper_tri_list_N[index_N]
#                 index_N += 1
#                 # 无论如何，我们都要增加 index_M_current 来移动到下一个位置
#             index_M_current += 1
#
#             # 此时，如果 N < M，则 upper_tri_list_M 的后半部分已经是0（因为我们初始化为0了）
#     # 如果 N == M，则所有元素都已经被正确映射，无需额外操作
#     return torch.Tensor(upper_tri_list_M)
#
# def map_upper_triangular_to_smaller_matrix(upper_tri_list_N, N, M, index_list):
#     Matrix = np.zeros([N, N])
#     ind = 0
#     for i in range(N):
#         for j in range(N):
#             if(j>i):
#                 Matrix[i, j] = upper_tri_list_N[ind]
#                 ind += 1
#
#     Matrix = Matrix[index_list, :]
#     Matrix = Matrix[:, index_list]
#     upper_tri_list_M = []
#     for i in range(M):
#         for j in range(M):
#             if(j>i):
#                 upper_tri_list_M.append(Matrix[i, j])
#     return torch.Tensor(upper_tri_list_M)