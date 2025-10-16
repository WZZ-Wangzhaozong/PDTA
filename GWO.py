import numpy as np
import torch
import random
import Env as Env
def fitness_calculation(Positions, Wolves_action, x0, end_symbol, opponent_model, loss_type):
    location = Wolves_action[0, :, 0]
    rec_time = Wolves_action[0, :, 2]
    action = Wolves_action[:, :, 3]

    Searchwolf_num = Positions.shape[0]
    P = Positions.shape[1] - 1
    reach_symbol = np.array([False for _ in range(Searchwolf_num)])

    if len(opponent_model) == 1:
        model = opponent_model[0]
    else:
        encoder = opponent_model[0]
        x0 = encoder(x0)
        model = opponent_model[1]

    if (loss_type == "Minimize Maximum"):
        with torch.no_grad():
            x0_tiled = x0.unsqueeze(0).repeat(Searchwolf_num, 1)

            # Calculating the anti-capabilities after the actions are applied
            for i in range(Wolves_action.shape[1]):
                # t_len: time duration in each step; iter_num: step number
                delta_t = rec_time[0] if i == 0 else rec_time[i] - rec_time[i - 1]
                iter_num = int(delta_t / 0.01) + 1
                t_len = delta_t / iter_num

                # if utilization of the agents is incomplete, adding the wasted resources to the fitness value
                if (torch.any(x0_tiled <= end_symbol)):
                    less_than_symbol = np.array(torch.all(x0_tiled <= end_symbol, dim=1))
                    indices = np.where((reach_symbol == False) & (less_than_symbol == True))[0]
                    Positions[indices, -1] -= (Wolves_action.shape[1] - i) * 1000000
                    reach_symbol[less_than_symbol] = True

                # if adversarial process is over, calculating the fitness
                if (torch.all(x0_tiled <= end_symbol)):
                    score2, _ = torch.max(x0_tiled, dim=1)
                    Positions[:, -1] += score2 * 100
                    return Positions[:, -1]

                # Recursion
                for j in range(int(iter_num)):
                    data1 = x0_tiled
                    data2 = torch.stack([data1 ** ind for ind in range(3)], dim=-1)
                    dxdt = model(data1, data2).reshape(Searchwolf_num, P)
                    x0_tiled += dxdt * t_len

                    # if utilization of the agents is incomplete, adding the wasted resources to the fitness value
                    if (torch.any(x0_tiled <= end_symbol)):
                        less_than_symbol = np.array(torch.all(x0_tiled <= end_symbol, dim=1))
                        indices = np.where((reach_symbol == False) & (less_than_symbol == True))[0]
                        Positions[indices, -1] -= (Wolves_action.shape[1] - i) * 1000000
                        reach_symbol[less_than_symbol] = True

                    # if adversarial process is over, calculating the fitness
                    if (torch.all(x0_tiled <= end_symbol)):
                        score2, _ = torch.max(x0_tiled, dim=1)
                        Positions[:, -1] += score2 * 100
                        return Positions[:, -1]

                # if utilization of the agents is incomplete, adding the wasted resources to the fitness value
                if (int(location[i]) <= x0_tiled.shape[1]):
                    x0_tiled[:, int(location[i]) - 1] -= action[:, i]
                    score1 = (x0_tiled[:, int(location[i]) - 1] - end_symbol)
                    score1 = torch.clamp((action[:, i] - score1), min=0.0)
                    Positions[:, -1] += score1 * 500000
                else:
                    score1 = action[:, i]
                    Positions[:, -1] += score1 * 500000

                # Setting anti-capabilities non-negative
                x0_tiled[x0_tiled < 0] = 0.0

        # calculating the fitness
        score2, _ = torch.max(x0_tiled, dim=1)
        Positions[:, -1] += score2 * 100
    elif (loss_type == "Maximize Decrease"):
        with torch.no_grad():
            x0_tiled1 = x0.unsqueeze(0).repeat(Searchwolf_num, 1)
            x0_tiled = x0.unsqueeze(0).repeat(Searchwolf_num, 1)
            for i in range(Wolves_action.shape[1]):
                # t_len: time duration in each step; iter_num: step number
                delta_t = rec_time[0] if i == 0 else rec_time[i] - rec_time[i - 1]
                iter_num = int(delta_t / 0.01) + 1
                t_len = delta_t / iter_num

                # if utilization of the agents is incomplete, adding the wasted resources to the fitness value
                if (torch.any(x0_tiled <= end_symbol)):
                    less_than_symbol = np.array(torch.all(x0_tiled <= end_symbol, dim=1))
                    indices = np.where((reach_symbol == False) & (less_than_symbol == True))[0]
                    Positions[indices, -1] -= (Wolves_action.shape[1] - i) * 1000000
                    reach_symbol[less_than_symbol] = True

                # if adversarial process is over, calculating the fitness
                if (torch.all(x0_tiled <= end_symbol)):
                    score2, _ = torch.max(x0_tiled, dim=1)
                    Positions[:, -1] += score2 * 100
                    return Positions[:, -1]

                # Recursion
                for j in range(int(iter_num)):
                    data1 = x0_tiled
                    data2 = torch.stack([data1 ** ind for ind in range(3)], dim=-1)
                    dxdt = model(data1, data2).reshape(Searchwolf_num, P)
                    x0_tiled += dxdt * t_len

                    # if utilization of the agents is incomplete, adding the wasted resources to the fitness value
                    if (torch.any(x0_tiled <= end_symbol)):
                        less_than_symbol = np.array(torch.all(x0_tiled <= end_symbol, dim=1))
                        indices = np.where((reach_symbol == False) & (less_than_symbol == True))[0]
                        Positions[indices, -1] -= (Wolves_action.shape[1] - i) * 1000000
                        reach_symbol[less_than_symbol] = True

                    # if adversarial process is over, calculating the fitness
                    if(torch.all(x0_tiled <= end_symbol)):
                        Positions[:, -1] += torch.sum(-x0_tiled1 + x0_tiled, axis=1) * 100
                        return Positions[:, -1]

                # if utilization of the agents is incomplete, adding the wasted resources to the fitness value
                if(int(location[i]) <= x0_tiled.shape[1]):
                    x0_tiled[:, int(location[i]) - 1] -= action[:, i]
                    score1 = (x0_tiled[:, int(location[i]) - 1] - end_symbol)
                    score1 = torch.clamp((action[:, i] - score1), min=0.0)
                    Positions[:, -1] += score1 * 500000
                else:
                    score1 = action[:, i]
                    Positions[:, -1] += score1 * 500000

                # Setting anti-capabilities non-negative
                x0_tiled[x0_tiled < 0] = 0.0

        # calculating the fitness
        Positions[:, -1] += torch.sum(-x0_tiled1 + x0_tiled, axis=1) * 100
    return Positions[:, -1]

def updating_wolves(Position, Alpha_wolf, Beta_wolf, Delta_wolf, Dim, a):
    for dim in range(Dim):
        r1 = random.random()  # r1 is a random number in [0,1]
        r2 = random.random()  # r2 is a random number in [0,1]
        A1 = 2 * a * r1 - a
        C1 = 2 * r2

        D_alpha = abs(C1 * Alpha_wolf[dim] - Position[dim])
        X1 = Alpha_wolf[dim] - A1 * D_alpha

        r1 = random.random()
        r2 = random.random()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = abs(C2 * Beta_wolf[dim] - Position[dim])
        X2 = Beta_wolf[dim] - A2 * D_beta

        r1 = random.random()
        r2 = random.random()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = abs(C3 * Delta_wolf[dim] - Position[dim])
        X3 = Delta_wolf[dim] - A3 * D_delta

        Position[dim] = (X1 + X2 + X3) / 3
    return Position

def DEGWO(Dim, Searchwolf_num, Max_iter, KO_action, x0, Time, end_symbol, unexcuted,
          to_be_instantiated, model, win_len, loss_type):
    # Initializing alpha, beta, and delta
    Alpha_wolf = torch.zeros(Dim); Beta_wolf = torch.zeros(Dim); Delta_wolf = torch.zeros(Dim)
    Alpha_score = float("inf"); Beta_score = float("inf"); Delta_score = float("inf")
    # Initializing all wolves
    Positions = torch.rand((Searchwolf_num, Dim + 1))
    lb = torch.zeros([Dim]); ub = torch.ones([Dim])

    # Iterative optimization
    for l in range(Max_iter):
        for dim in range(Dim):
            Positions[:, dim] = torch.clamp(Positions[:, dim], min=lb[dim], max=ub[dim]) + 1e-6
        for i in range(Searchwolf_num):
            Positions[i, :Dim] /= torch.sum(Positions[i, :Dim])
        Positions[:, -1] = torch.zeros([Searchwolf_num])

        # Wolves_action is [Searchwolf_num * action_num * [Loc, ret_time, rec_time, u]]
        # para = [Searchwolf_num, Dim, KO_action, win_len] + Time.tolist() + x0.tolist() + [end_symbol] + Positions[:, :Dim].flatten().tolist()
        para = [Searchwolf_num, Dim, KO_action] + Time.tolist() + x0.tolist() + [end_symbol] + Positions[:, :Dim].flatten().tolist()
        Wolves_action = Env.task_symbol_Csharp(list(map(str, para)), unexcuted, to_be_instantiated, Searchwolf_num, Dim)
        Positions[:, -1] = fitness_calculation(Positions, Wolves_action, x0, end_symbol, model, loss_type)

        # Sorting...
        sort_indices = torch.argsort(Positions[:, -1])
        Positions = Positions[sort_indices, :]

        # Updating alpha, beta, and delta
        for i in range(3):
            fitness = Positions[i, -1]
            if (fitness < Alpha_score):
                Alpha_score = fitness.clone()  # Update alpha
                Alpha_wolf = Positions[i, :-1].clone()
            elif (fitness > Alpha_score and fitness < Beta_score):
                Beta_score = fitness.clone()  # Update beta
                Beta_wolf = Positions[i, :-1].clone()
            elif (fitness > Beta_score and fitness < Delta_score):
                Delta_score = fitness.clone()  # Update delta
                Delta_wolf = Positions[i, :-1].clone()

        # Updating wolves according to alpha, beta, and delta
        a = 2 - l * (2 / Max_iter)  # a decreases linearly from 2 to 0
        for wolf in range(Searchwolf_num):
            Positions[wolf, :Dim] = updating_wolves(Positions[wolf, :Dim], Alpha_wolf, Beta_wolf, Delta_wolf, Dim, a)

    return Alpha_wolf/sum(Alpha_wolf), Alpha_score