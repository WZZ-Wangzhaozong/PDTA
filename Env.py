import numpy as np
from scipy.integrate import solve_ivp
import torch
import subprocess
import json
import sys
import os

def forward_equations(t, x, KE, KR, PNi, Bij, exp, beta, end_symbol):
    '''
    :param t, x, KE, KR, PNi, Bij, exp, beta: input
    :return: current change state trends of opponents, used for predicting the future state through integration (during adversarial process)
    '''
    x = torch.Tensor(x).clone()
    dim = x.shape[0]

    for i in range(dim):
        x[i] = max(0, x[i])
    x_nonnegative = torch.max(x, torch.zeros_like(x))

    dxdt1 = torch.zeros_like(x)
    for i in range(dim):
        for j in range(dim):
            term = torch.sign(x_nonnegative[j] / PNi[j] - x_nonnegative[i] / PNi[i]) * \
                   (exp ** torch.abs(x_nonnegative[j] / PNi[j] - x_nonnegative[i] / PNi[i]) - 1) / \
                   Bij[i, j]
            dxdt1[i] += term
    dxdt1 *= KE

    dxdt2 = (beta[0] + beta[1] * x_nonnegative + beta[2] * (x_nonnegative ** 2)) * KR
    return dxdt1 + dxdt2

def backward_equations(x, KE, KR, PNi, Bij, exp, beta):
    '''
    :param x, KE, KR, PNi, Bij, exp, beta: input
    :return: current change state trends of opponents, used for saving data (after adversarial process)
    '''
    dim = x.shape[0]-1
    dxdt1 = np.zeros([1+dim**2])
    dxdt2 = np.zeros([1+dim])
    dxdt1[0] = x[0]
    dxdt2[0] = x[0]
    x = x[1:]

    for i in range(dim):
        for j in range(dim):
            dxdt1[i * dim + j + 1] = np.sign(x[j] / PNi[j] - x[i] / PNi[i]) * (
                        exp ** abs(x[j] / PNi[j] - x[i] / PNi[i]) - 1) / Bij[i, j]
    dxdt1[1:] *= KE

    dxdt2[1:] = (beta[0] + x * beta[1] + x ** 2 * beta[2]) * KR
    return dxdt1, dxdt2

def Process(args, len, x0):
    if(len==0.0):
        return np.array([0.0]), x0.reshape(1, -1)
    t_span = [0, len]
    t_eval = np.linspace(t_span[0], t_span[1], max(100, int(len*100)+1))   # 评估时间点

    sol = solve_ivp(forward_equations, t_span, x0, t_eval=t_eval, args=args)
    return sol.t, sol.y.T

def task_symbol(P, Time, Win_len, tol=0.05):
    task_pending = np.array([[i+1, Time[i] * 2, Time[i], 0.0] for i in range(P)]).T
    task_performed = np.empty((4, 0))

    while(True):
        # task_pending = task_pending.T[np.argsort(task_pending.T[:, 1])].T
        task_pending = task_pending[:, np.argsort(task_pending[1])]
        if(task_pending[1, 0] > Win_len):
            break

        time_pen = task_pending[1, 0]
        task_performed = np.hstack([task_performed, task_pending[:, [0]]])
        task_pending = np.delete(task_pending, [0], axis=1)
        new_tasks = np.array([
            [i+1, time_pen + Time[i] * 2, time_pen + Time[i], 0.0]
            for i in range(P)]).T
        task_pending = np.hstack([task_pending, new_tasks])

    ind = []
    unique_vals = []
    unique_vals1 = []
    sums = []

    for i in range(task_performed.shape[1]):
        found = False
        for j in range(len(unique_vals)):
            if np.abs(task_performed[1, i] - unique_vals[j]) <= tol and task_performed[0, i] == ind[j]:
                sums[j] += task_performed[3, i]
                found = True
                break
        if not found:
            ind.append(task_performed[0, i])
            unique_vals.append(task_performed[1, i])
            unique_vals1.append(task_performed[2, i])
            sums.append(task_performed[3, i])

    merged_matrix = np.zeros((4, len(unique_vals)))
    merged_matrix[0] = ind
    merged_matrix[1] = unique_vals
    merged_matrix[2] = unique_vals1
    merged_matrix[3] = sums
    # merged_matrix = np.vstack([merged_matrix, np.ones(merged_matrix.shape[1])])
    merged_matrix = merged_matrix[:, np.argsort(merged_matrix[2, :])]
    return torch.Tensor(merged_matrix)

def task_symbol_Csharp(param, unexcuted, tobe_confirmed, Searchwolf_num, P):
    csharp_program_path = os.path.dirname(sys.path[0]) + "\C_sharp\Action_alg_P="+str(P)+'\\Action_alg_P='\
                          +str(P)+'\\bin\\Debug\\Action_alg_P='+str(P)+'.exe'

    try:
        command = [csharp_program_path] + param
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode == 0:
            data = json.loads(result.stdout)
        else:
            print("Error executing C# program:", result.stderr)
    except Exception as e:
        print(f"An error occurred while running the C# program: {e}")

    action_list = []
    for i in range(Searchwolf_num):
        task_ins = tobe_confirmed.clone()
        task_ins[-1, :] = torch.Tensor(data[i])
        task_ins = torch.cat((task_ins, unexcuted), dim=1)
        sort_indices = torch.argsort(task_ins[2, :])
        task_ins = task_ins[:, sort_indices]
        action_list.append(task_ins.T)
    return torch.stack(action_list, dim=0)

def Env_recursion(cur_time, act_time, act_seq, KO, anti_abilities, opponent_his_data, args):
    # if act_seq is not empty
    act_seq = act_seq[:, np.argsort(act_seq[2, :])]  # Sort action sequence according to arrival time
    while (act_seq.shape[1] != 0):
        recurs_time = act_seq[2, 0]  # Current moment
        index = np.where(act_seq[2, :] == recurs_time)[0]  # Index of actions at current moment
        act = act_seq[:, index]  # Actions at current moment

        # Calculate data between start and current time
        process_time, process_data = Process(args, recurs_time-cur_time, anti_abilities)
        opponent_cur_data = np.hstack([(process_time+cur_time).reshape(-1, 1), process_data])

        # Merge data between start and current time into history data
        opponent_his_data = np.vstack([opponent_his_data, opponent_cur_data])

        # Action effect and limit anti-capability non-negative
        anti_abilities = process_data[-1, :]
        for i in range(act.shape[1]):
            if(int(act[0, i]) <= anti_abilities.shape[0]):
                anti_abilities[int(act[0, i]) - 1] -= KO * act[3, i]
                anti_abilities[int(act[0, i]) - 1] = max(0.0, anti_abilities[int(act[0, i]) - 1])
        opponent_his_data[-1, 1:] = anti_abilities

        cur_time = recurs_time
        act_seq = np.delete(act_seq, index, axis=1)

        # The confrontation is over
        if(np.all(anti_abilities <= args[-1])):
            return torch.Tensor(anti_abilities), opponent_his_data

    # elif act_seq is empty, only used to recursion without action sequence
    process_time, process_data = Process(args, act_time-cur_time, anti_abilities)
    opponent_cur_data = np.hstack([(process_time+cur_time).reshape(-1, 1), process_data])
    opponent_his_data = np.vstack([opponent_his_data, opponent_cur_data])
    anti_abilities = torch.Tensor(process_data[-1, :])

    return anti_abilities, opponent_his_data

def merge_columns_with_tolerance(matrix, tol=0.05):
    '''
    :param matrix: Action sequence, elements: [time, action]
    :param tol: Merging the elements that occur within "tol" seconds of each other
    :return: Merged action sequence
    '''
    # original version
    # unique_vals = []
    # sums = []
    #
    # for i in range(matrix.shape[1]):
    #     found = False
    #     for j in range(len(unique_vals)):
    #         if np.abs(matrix[0, i] - unique_vals[j]) <= tol:
    #             sums[j] += matrix[1, i]
    #             found = True
    #             break
    #     if not found:
    #         unique_vals.append(matrix[0, i])
    #         sums.append(matrix[1, i])
    #
    # merged_matrix = np.zeros((2, len(unique_vals)))
    # merged_matrix[0] = unique_vals
    # merged_matrix[1] = sums
    # return merged_matrix

    # new version
    sorted_indices = np.argsort(matrix[0])
    times = matrix[0][sorted_indices]
    actions = matrix[1][sorted_indices]

    merged_times = []
    merged_actions = []

    current_time = times[0]
    current_sum = actions[0]

    for i in range(1, len(times)):
        if times[i] - current_time <= tol:
            current_sum += actions[i]
        else:
            merged_times.append(current_time)
            merged_actions.append(current_sum)
            current_time = times[i]
            current_sum = actions[i]

    merged_times.append(current_time)
    merged_actions.append(current_sum)
    return np.array([merged_times, merged_actions])