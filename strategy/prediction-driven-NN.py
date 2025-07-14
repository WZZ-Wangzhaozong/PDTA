import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time as T
import NN.NN_model as NN_model
import Env as Env
import GWO as GWO
import os
import parameter
from collections import defaultdict
from multiprocessing import Pool
import copy
import sys


class Data_save:
    @staticmethod
    def Sampling(opponent_his_data, given_intervals=0.05):
        sample_time = 0
        index_list = [0]
        begin = opponent_his_data[0, 0]
        end = opponent_his_data[-1, 0]
        while (True):
            if ((begin + given_intervals * (sample_time + 1)) > end):
                index_list.append(-1)
                break
            else:
                index = np.where(opponent_his_data[:, 0] <= (begin + given_intervals * (sample_time + 1)))[0][-1]
                index_list.append(index)
                sample_time += 1
        return opponent_his_data[index_list, :]

    @staticmethod
    def Calculate_velocity(P, opponent_his_data, KE, KR, PNi, Bij, exp, beta):
        opponent_exc_vel_data = np.empty([0, 1 + P ** 2])
        opponent_rec_vel_data = np.empty([0, 1 + P])
        for i in range(opponent_his_data.shape[0]):
            dxdt1, dxdt2 = Env.backward_equations(opponent_his_data[i, :], KE, KR, PNi, Bij, exp, beta)
            opponent_exc_vel_data = np.vstack([opponent_exc_vel_data, dxdt1])
            opponent_rec_vel_data = np.vstack([opponent_rec_vel_data, dxdt2])
        return opponent_exc_vel_data, opponent_rec_vel_data

    @staticmethod
    def Calculate_amount(opponent_exc_vel_data, opponent_rec_vel_data):
        opponent_exc_data = opponent_exc_vel_data.copy()
        opponent_rec_data = opponent_rec_vel_data.copy()
        for i in range(1, opponent_exc_data.shape[0]):
            opponent_exc_data[i, 1:] = \
                opponent_exc_data[i, 1:] * (
                        opponent_exc_data[i, 0] - opponent_exc_data[i - 1, 0]) + opponent_exc_data[
                                                                                 i - 1, 1:]
        for i in range(1, opponent_exc_data.shape[0]):
            opponent_rec_data[i, 1:] = \
                opponent_rec_data[i, 1:] * (
                        opponent_rec_data[i, 0] - opponent_rec_data[i - 1, 0]) + opponent_rec_data[
                                                                                 i - 1, 1:]
        return opponent_exc_data, opponent_rec_data

    @staticmethod
    def Save_data_as_npy(save_path, file_name, opponent_his_data, opponent_exc_data, opponent_rec_data,
                         opponent_exc_vel_data, opponent_rec_vel_data, action_sequence, Ass_res_his):
        Adv_phase, KE, KR, KO = file_name
        np.save(
            save_path + "opponents_Adv_phase=" + str(Adv_phase - 1) + "_KE=" + str(KE) + "_KR=" + str(KR)
            + "_KO=" + str(KO) + ".npy", opponent_his_data)

        np.save(
            save_path + "opponent_exc_Adv_phase=" + str(Adv_phase - 1) + "_KE=" + str(KE) + "_KR=" + str(
                KR)
            + "_KO=" + str(KO) + ".npy", opponent_exc_data)

        np.save(
            save_path + "opponent_rec_Adv_phase=" + str(Adv_phase - 1) + "_KE=" + str(KE) + "_KR=" + str(
                KR)
            + "_KO=" + str(KO) + ".npy", opponent_rec_data)

        np.save(save_path + "opponent_exc_vel_Adv_phase=" + str(Adv_phase - 1) + "_KE=" + str(
            KE) + "_KR=" + str(KR)
                + "_KO=" + str(KO) + ".npy", opponent_exc_vel_data)

        np.save(save_path + "opponent_rec_vel_Adv_phase=" + str(Adv_phase - 1) + "_KE=" + str(
            KE) + "_KR=" + str(KR)
                + "_KO=" + str(KO) + ".npy", opponent_rec_vel_data)

        np.save(save_path + "act_seq_Adv_phase=" + str(Adv_phase - 1) + "_KE=" + str(KE) + "_KR=" + str(KR)
                + "_KO=" + str(KO) + ".npy", action_sequence)

        np.save(save_path + "Ass_res_his_Adv_phase=" + str(Adv_phase - 1) + "_KE=" + str(KE) + "_KR=" + str(KR)
                + "_KO=" + str(KO) + ".npy", Ass_res_his)


class Initialize_each_phase:
    @staticmethod
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

    @staticmethod
    def Prediction_duration_select(P, Time):
        '''
        :param P: Opponents' number
        :param Time: Travel times to opponents
        :return: Prediction_duration t_len
        '''
        if (P <= 4):
            return 6.0 + 48.2755144334742 / 62.08641773198754
        elif (P <= 6):
            return 3.5 + 30.157475345595586 / 62.08641773198754
        else:
            return 3.0 + 23.665621261737517 / 38.304951684997060

    @staticmethod
    def loss_type_select(PNi, Bij, tol_PNi=0.1, tol_Bij=10000):
        '''
        :param PNi: Importance ϵ
        :param Bij: Transfer damping
        :param tol_PNi: Variance tolerated to importance
        :param tol_Bij: Variance tolerated to transfer damping
        :return: Objective function type
        '''
        PNi = 1 / PNi
        PNi = PNi / np.sum(PNi) * PNi.shape[0]

        if(np.std(PNi) > tol_PNi):
            return "Maximize Decrease"
        elif np.any(Bij) > tol_Bij:
            return "Maximize Decrease"
        return "Minimize Maximum"


class Model_fine_tuning:
    @staticmethod
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

    # @staticmethod
    # def load_encoder_decoder(retrain, P, P_last_phase, phase):
    #     if(retrain):
    #         if(phase == 0):
    #             ae1 = NN_model.DiagonalLinear(channels=P)
    #             ae2 = NN_model.DiagonalLinear(channels=P)
    #             ae1.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae1_Adv_phase=" + str(phase) + ".pkl"))
    #             ae2.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae2_Adv_phase=" + str(phase) + ".pkl"))
    #         else:
    #             ae1_old = NN_model.DiagonalLinear(channels=P_last_phase)
    #             ae2_old = NN_model.DiagonalLinear(channels=P_last_phase)
    #             ae1_old.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae1_Adv_phase=" + str(phase-1) + ".pkl"))
    #             ae2_old.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae2_Adv_phase=" + str(phase-1) + ".pkl"))
    #             ae1 = NN_model.DiagonalLinear(channels=P)
    #             ae2 = NN_model.DiagonalLinear(channels=P)
    #
    #             if P >= P_last_phase:
    #                 print(phase)
    #                 ae1.diagonal[:P_last_phase] = ae1_old.diagonal
    #                 ae2.diagonal[:P_last_phase] = ae2_old.diagonal
    #             else:
    #                 preserve_index = [parameter.opponents_index[phase-1].index(x) for x in
    #                                   parameter.opponents_index[phase]]
    #                 ae1.diagonal = ae1_old.diagonal[preserve_index]
    #                 ae2.diagonal = ae2_old.diagonal[preserve_index]
    #     else:
    #         ae1 = NN_model.DiagonalLinear(channels=P)
    #         ae2 = NN_model.DiagonalLinear(channels=P)
    #         ae1.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae1_Adv_phase=" + str(phase) + ".pkl"))
    #         ae2.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\ae2_Adv_phase=" + str(phase) + ".pkl"))
    #     return ae1, ae2

    @staticmethod
    def load_encoder_decoder(P, phase):
        encoder = NN_model.DiagonalLinear(channels=P)
        decoder = NN_model.DiagonalLinear(channels=P)
        coder_path = os.path.dirname(sys.path[0]) + r"\NN\trained_NN"
        encoder.load_state_dict(torch.load(coder_path + r"\ae1_Adv_phase=" + str(phase) + ".pkl"))
        decoder.load_state_dict(torch.load(coder_path + r"\ae2_Adv_phase=" + str(phase) + ".pkl"))
        return encoder, decoder

    @staticmethod
    def original_transfer_new_coder(retrain, P, P_last_phase, phase):
        if (retrain):
            if (phase == 0):
                encoder, decoder = Model_fine_tuning.load_encoder_decoder(P, phase)
            else:
                encoder_original, decoder_original = Model_fine_tuning.load_encoder_decoder(P_last_phase, phase-1)
                encoder = NN_model.DiagonalLinear(channels=P)
                decoder = NN_model.DiagonalLinear(channels=P)

                # Retain the parameters in last model and extend them to the new one
                if P >= P_last_phase:
                    new_diag1 = torch.ones(P, device=encoder_original.diagonal.device)
                    new_diag2 = torch.ones(P, device=decoder_original.diagonal.device)
                    new_diag1[:P_last_phase] = encoder_original.diagonal
                    new_diag2[:P_last_phase] = decoder_original.diagonal
                    encoder.diagonal = nn.Parameter(new_diag1)
                    decoder.diagonal = nn.Parameter(new_diag2)
                else:
                    preserve_index = [parameter.opponents_index[phase - 1].index(x) for x in
                                      parameter.opponents_index[phase]]
                    new_diag1 = encoder_original.diagonal[preserve_index]
                    new_diag2 = decoder_original.diagonal[preserve_index]
                    encoder.diagonal = nn.Parameter(new_diag1)
                    decoder.diagonal = nn.Parameter(new_diag2)
        else:
            encoder, decoder = Model_fine_tuning.load_encoder_decoder(P, phase)
        return encoder, decoder

    @staticmethod
    def load_model(P, order, phase):
        model = NN_model.Opponent_model_explicit(channels=P, order=order)
        model.load_state_dict(torch.load(model_path + str(phase) + ".pkl"))
        return model

    @staticmethod
    def original_transfer_new(retrain, P, P_last_phase, Adv_phase):
        opponents_number_change = P - P_last_phase
        if (not retrain):
            print("Phase " + str(Adv_phase) + " opponent model being used")
            model = Model_fine_tuning.load_model(P, order, Adv_phase)
            Model_fine_tuning.print_parameters_in_model(model)
        else:
            print("Phase " + str(Adv_phase-1) + " opponent model being transferred")
            if (opponents_number_change == 0):
                model = Model_fine_tuning.load_model(P, order, Adv_phase-1)
            else:
                model_ori = Model_fine_tuning.load_model(P_last_phase, order, Adv_phase-1)
                model = NN_model.Opponent_model_explicit(channels=P, order=order)

                # Transfer parameters in old network to new network
                with torch.no_grad():
                    # Recovery network transfer
                    model.Son_recovernet.fc.load_state_dict(model_ori.Son_recovernet.fc.state_dict())

                    # IE network transfer
                    indices = [parameter.opponents_index[Adv_phase-1].index(x) for x in parameter.opponents_index[Adv_phase]]
                    if (opponents_number_change < 0):
                        model.Son_swapnet.fc1.diagonal.data.copy_(model_ori.Son_swapnet.fc1.diagonal.data[indices])
                    else:
                        model.Son_swapnet.fc1.diagonal.data[:P_last_phase].copy_(model_ori.Son_swapnet.fc1.diagonal.data)
                        model.Son_swapnet.fc1.diagonal.data[P_last_phase:] = model_ori.Son_swapnet.fc1.diagonal.data.sum().item() / P_last_phase

                    # BC network transfer
                    model.Son_swapnet.fc2.load_state_dict(model_ori.Son_swapnet.fc2.state_dict())

                    # TD network transfer
                    for parameters in model_ori.Son_swapnet.fc3.parameters():
                        para = torch.Tensor(parameters)
                        upper_tri_list_M = Model_fine_tuning.TD_layer_transfer(para, P_last_phase, P, indices)
                        model.Son_swapnet.fc3.diagonal = nn.Parameter(upper_tri_list_M)
            print("Phase " + str(Adv_phase-1) + " opponent model has been transferred to the one in phase " + str(Adv_phase))
        return model
    
    @staticmethod
    def TD_layer_transfer(upper_tri_list_N, N, M, index_list):
        if (N > M):
            Matrix = np.zeros([N, N])
            ind = 0
            for i in range(N):
                for j in range(N):
                    if (j > i):
                        Matrix[i, j] = upper_tri_list_N[ind]
                        ind += 1

            Matrix = Matrix[index_list, :]
            Matrix = Matrix[:, index_list]
            upper_tri_list_M = []
            for i in range(M):
                for j in range(M):
                    if (j > i):
                        upper_tri_list_M.append(Matrix[i, j])
            return torch.Tensor(upper_tri_list_M)

        elif (N < M):
            num_elements_N = (N * (N - 1)) // 2

            if len(upper_tri_list_N) != num_elements_N:
                raise ValueError("The given list length does not match the number of triangular elements on the N×N matrix")

            num_elements_M = (M * (M - 1)) // 2
            upper_tri_list_M = [1] * num_elements_M

            index_N = 0
            index_M_current = 0
            for i in range(M):
                for j in range(i + 1, M):
                    if i < N and j < N:
                        upper_tri_list_M[index_M_current] = upper_tri_list_N[index_N]
                        index_N += 1
                    index_M_current += 1

            return torch.Tensor(upper_tri_list_M)

    # @staticmethod
    # def train_multiple_models_mp(train_loader, model, num_epochs, trainable_layers):
    #     trainable_layers = [['fc1'], ['fc3'], ['ae1'], ['fc1', 'fc3'], ['fc1', 'ae1'], ['fc3', 'ae1'],
    #                         ['fc1', 'fc3', 'ae1'], ['all']]
    #
    #     parallel_fine_tune_num = len(trainable_layers)
    #     model_copies = [copy.deepcopy(model) for _ in range(parallel_fine_tune_num)]
    #     args_list = [(train_loader, model_copies[i], num_epochs, trainable_layers[i])
    #                  for i in range(parallel_fine_tune_num)]
    #
    #     with Pool(processes=parallel_fine_tune_num) as pool:
    #         results = pool.starmap(NN_model.model_fine_tune, args_list)
    #     return zip(*results)

    @staticmethod
    def similarity_calculation(test_loader, target_model, current_model):
        for i in range(len(target_model)):
            target_model[i].eval()
            current_model[i].eval()

        total_relative_error = 0.0
        total_samples = 0

        with torch.no_grad():
            if(len(target_model) == 1):  # Without deception of opponents
                for inputs1, inputs2, _ in test_loader:
                    outputs_model1 = target_model[0](inputs1, inputs2)
                    outputs_model2 = current_model[0](inputs1, inputs2)

                    bias = outputs_model1 - outputs_model2
                    eps = 1e-8  # Prevent division by 0
                    relative_error = torch.abs(bias) / (torch.abs(outputs_model1) + eps)

                    total_relative_error += relative_error.sum()
                    total_samples += relative_error.numel()
            else:  # Existing deception of opponents
                for inputs, _ in test_loader:
                    inputs = inputs.float()

                    input_explicit = target_model[0](inputs)
                    input_explicit_pow = torch.stack([input_explicit ** i for i in range(target_model[1].order)], dim=-1)
                    output_explicit = target_model[1](input_explicit, input_explicit_pow)[:, :, 0]
                    outputs_model1 = target_model[2](output_explicit)

                    input_explicit = current_model[0](inputs)
                    input_explicit_pow = torch.stack([input_explicit ** i for i in range(current_model[1].order)], dim=-1)
                    output_explicit = current_model[1](input_explicit, input_explicit_pow)[:, :, 0]
                    outputs_model2 = current_model[2](output_explicit)

                    bias = outputs_model1 - outputs_model2
                    eps = 1e-8  # Prevent division by 0
                    relative_error = torch.abs(bias) / (torch.abs(outputs_model1) + eps)

                    total_relative_error += relative_error.sum()
                    total_samples += relative_error.numel()

        final_mean_relative_error = total_relative_error / total_samples
        similarity = 1 / (1 + final_mean_relative_error)
        return similarity

    @staticmethod
    def fidelity_calculation(test_loader, model):
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs1, inputs2, targets in test_loader:
                outputs, _, _ = model(inputs1, inputs2)
                loss = criterion(outputs.reshape(-1, targets.shape[1]), targets)
                total_loss += loss.item() * inputs1.size(0)
                total_samples += inputs1.size(0)

        return total_loss / total_samples


if __name__ == '__main__':
    '''GWO Hyperparameters'''
    Searchwolf_num = parameter.Searchwolf_num
    Max_iter = parameter.Max_iter

    '''Confrontation Start, Process and End Settings'''
    end_symbol = parameter.end_symbol
    Adv_time = parameter.Adv_time
    anti_capability = parameter.x0  # Anti-capability sequence of opponents
    
    '''Allies and Opponents Capability Parameters'''
    KO = parameter.KO
    Agent_number = parameter.Agent_number
    KE = parameter.KE
    exp = parameter.exp
    KR = parameter.KR
    beta = parameter.beta
    order = parameter.order

    '''Adversarial process initialization'''
    # 2. Initializing adversarial process
    Ass_res_his = np.zeros([4, 20000])  # Ass_res_his is used to store confirmed assignment results, including:
    # [opponent index, time returning to base, time arriving at opponent, agent ID / total agents]
    s1_len = 0  # Length of valid information in Ass_res_his

    Agent_tobe_ass = np.zeros([2, 20000])  # Agent_tobe_ass is used to store agents awaiting assignment, including:
    # [departure time from base, agent ID / total agents]
    Agent_tobe_ass[:, 0] = np.array([0.0, 1.0])
    s2_len = 1  # Length of valid information in Agent_tobe_ass

    adv_cur_time = 0.0
    Adv_phase = 0  # Adversarial phase
    train_time = 0.0

    retrain = False
    deception = False
    strategy = "prediction-driven-NN"  # "Quantity-based"  # "AST", "TASE", "prediction-driven", "prediction-NN-driven"

    if strategy == "prediction-driven-NN":
        retrain = True
    elif strategy not in ["Quantity-based", "AST", "TASE", "prediction-driven"]:
        print("Please check whether the strategy name is correct...")
        sys.exit()

    model_path = os.path.dirname(sys.path[0]) + r"\NN\trained_NN\Adv_phase="
    data_path = os.path.dirname(sys.path[0]) + r"\data_save\process_data" + "\\" + strategy + "\\"
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    '''Print network parameters for all phases in order to check'''
    if retrain:
        print("Model in phase " + str(-1) + ":")
        model = Model_fine_tuning.load_model(parameter.P[0], order, -1)
        Model_fine_tuning.print_parameters_in_model(model)
    for phase in range(len(Adv_time)):
        print("Model in phase " + str(phase) + ":")
        model = Model_fine_tuning.load_model(parameter.P[phase], order, phase)
        Model_fine_tuning.print_parameters_in_model(model)

    while (torch.any(anti_capability > end_symbol)):  # Sign of the end of the adversarial process
        if(Adv_phase < len(Adv_time) and adv_cur_time >= Adv_time[Adv_phase]):  # The opponent strategy changes and the adversarial process enters a new phase
            print("Entering confrontation phase " + str(Adv_phase) + "...")
            print(" ")
            if(Adv_phase >= 1):
                # Sample the adversarial process data at intervals of 0.05 seconds.
                opponent_his_data = Data_save.Sampling(opponent_his_data, given_intervals=0.05)
                # Calculate the exchange and recovery velocity of opponents.
                opponent_exc_vel_data, opponent_rec_vel_data = Data_save.Calculate_velocity(P, opponent_his_data, KE, KR, PNi, Bij, exp, beta)
                # Calculate the exchange and recovery amount of opponents.
                opponent_exc_data, opponent_rec_data = Data_save.Calculate_amount(opponent_exc_vel_data, opponent_rec_vel_data)
                # Save data as npy.
                Data_save.Save_data_as_npy(data_path, [Adv_phase, KE, KR, KO], opponent_his_data, opponent_exc_data, opponent_rec_data,
                                 opponent_exc_vel_data, opponent_rec_vel_data, action_sequence, Ass_res_his)

            P_last_phase = parameter.P[Adv_phase-1] if Adv_phase > 0 else parameter.P[0]  # Number of opponents in last phase
            P = parameter.P[Adv_phase]  # Updating the number of opponents
            opponents_number_change = P - P_last_phase  # Change in number of opponents
            opponents_index = parameter.opponents_index[Adv_phase]  # Opponent index
            Time = np.array(parameter.Time)[opponents_index]  # Updating travel time from base to opponents
            PNi = parameter.PNi[Adv_phase]  # Updating the importance of opponents
            DLi = np.array(parameter.DLi[Adv_phase], dtype=np.float32)  # Updating the deception level of opponents
            Connect_graph = parameter.Connect_graph[Adv_phase]  # Updating connectivity of opponents
            Bij = parameter.Bij[np.ix_(parameter.opponents_index[Adv_phase], parameter.opponents_index[Adv_phase])]  # Bij considering solely the path length
            Bij = Initialize_each_phase.mask_matrix_by_labels(Bij, Connect_graph)  # Updating Bij (Bij considering the path length and connectivity)
            args = (KE, KR, PNi, Bij, exp, beta, end_symbol)

            prediction_horizon = Initialize_each_phase.Prediction_duration_select(P, Time)
            task = Env.task_symbol(P, Time, prediction_horizon)  # Instantiate tasks within prediction horizon, including:
            # [opponent index, time returning to base, time arriving at opponent, assigned ability (to be instantiated)]
            loss_type = Initialize_each_phase.loss_type_select(PNi, Bij)  # Object_function, Eq.(24)
            # if(Adv_phase==0):
            #     loss_type = "Minimize Maximum"
            # else:
            #     loss_type = "Maximize Decrease"

            # Generate a new round of data containers
            opponent_his_data = np.empty([0, P+1])
            action_sequence = np.empty([0, P+1])

            # Updating anti-capabilities if the opponents' number varies
            if(opponents_number_change > 0):
                x0_new = torch.full((opponents_number_change, ), 2.0)
                anti_capability = torch.cat((anti_capability, x0_new), dim=0)
            elif(opponents_number_change < 0):
                preserve_index = [parameter.opponents_index[Adv_phase-1].index(x) for x in opponents_index]
                abandon_index = [i for i, x in enumerate(parameter.opponents_index[Adv_phase-1]) if x not in opponents_index]
                preserved_values = anti_capability[preserve_index]
                adjustment = anti_capability[abandon_index].sum() / preserved_values.shape[0]
                anti_capability = preserved_values + adjustment

            # Recursing the environment from the current time to the next action moment
            adv_cur_time = Adv_time[Adv_phase]
            next_act_time = Agent_tobe_ass[0, 0].item()
            anti_capability, opponent_his_data = Env.Env_recursion(adv_cur_time, next_act_time,
                                                                   act_seq=np.empty([4, 0]), KO=None,
                                                                   anti_abilities=anti_capability,
                                                                   opponent_his_data=opponent_his_data,
                                                                   args=args)
            adv_cur_time = Agent_tobe_ass[0, 0].item()

            retrain = True if strategy in ["prediction-driven-NN", "prediction-driven-NN_deception"] else False

            model_copies = []
            model = Model_fine_tuning.original_transfer_new(retrain, P, P_last_phase, Adv_phase)
            encoder, decoder = Model_fine_tuning.original_transfer_new_coder(retrain, P, P_last_phase, Adv_phase)
            opponent_model = [encoder, model, decoder] if deception else [model]

            target_model = Model_fine_tuning.load_model(P, order, Adv_phase)
            target_encoder, target_decoder = Model_fine_tuning.load_encoder_decoder(P, Adv_phase)
            target_opponent_model = [target_encoder, target_model, target_decoder] if deception else [target_model]

            Adv_phase += 1  # Update phase number
            retrain_number = 0  # Update retraining number

        # Sorting the order of confirmed assignment results according to the moment of arrival at the opponent
        sort = np.argsort(Ass_res_his[2, :s1_len])
        Ass_res_his[:, :s1_len] = Ass_res_his[:, sort]

        # Prioritization of agents awaiting assignment according to the moment of return to the base
        sort = np.argsort(Agent_tobe_ass[0, :s2_len])
        Agent_tobe_ass[:, :s2_len] = Agent_tobe_ass[:, sort]

        # Merging the elements in Agent_tobe_ass that occur within 0.05 seconds of each other
        merged_matrix = Env.merge_columns_with_tolerance(Agent_tobe_ass[:, :s2_len], tol=0.05)
        s2_len = merged_matrix.shape[1]
        Agent_tobe_ass[:, :s2_len] = merged_matrix

        # Extracting the tasks that have been determined but not executed
        adv_cur_time = Agent_tobe_ass[0, 0].item()
        cols = np.where(Ass_res_his[2, :s1_len] > adv_cur_time)[0]
        Ass_res_his_current_round = Ass_res_his[:, cols.tolist()].copy()
        Ass_res_his_current_round[2, :] -= adv_cur_time

        # Calculating Weakening ability
        KO_action = KO * Agent_tobe_ass[1, 0].item()  # Agent_tobe_ass[1, 0] represents the number of available agents / total number
        print("Actual Capability:" + str(KO_action))
        Agent_tobe_departed_number = int(round(Agent_tobe_ass[1, 0].item() * Agent_number))  # the number of available agents
        print("Number of agents in this batch:" + str(Agent_tobe_departed_number))

        # Adding deception component to the observed anti-capabilities
        opponent_observed = anti_capability * DLi if deception else anti_capability

        s1 = T.time()
        if (strategy == "Quantity-based"):
            print("Optimization Object:Quantity-based")
            u_opt = (opponent_observed / sum(opponent_observed)).detach().numpy()
        elif (strategy == "AST"):
            print("Optimization Object:AST")
            result = [y / x for x, y in zip(Time, opponent_observed)]
            u_opt = result / np.sum(result)
        elif (strategy == "TASE"):
            print("Optimization Object:Minimize Maximum")
            u_opt, score = GWO.DEGWO(P, Searchwolf_num, Max_iter, KO_action, opponent_observed, Time, end_symbol,
                                     torch.Tensor(Ass_res_his_current_round), task, opponent_model, prediction_horizon,
                                     loss_type="Minimize Maximum")
            u_opt = u_opt.detach().numpy()
        elif (strategy == "TNNLS"):
            print("Optimization Object:Minimize Maximum")
            u_opt, score = GWO.DEGWO(P, Searchwolf_num, Max_iter, KO_action, opponent_observed, Time, end_symbol,
                                     torch.Tensor(Ass_res_his_current_round), task, opponent_model, prediction_horizon,
                                     loss_type="Minimize Maximum")
            u_opt = u_opt.detach().numpy()
        elif (strategy == "prediction-driven" or strategy == "prediction-driven-NN"):
            print("Optimization Object:" + str(loss_type))
            u_opt, score = GWO.DEGWO(P, Searchwolf_num, Max_iter, KO_action, opponent_observed, Time, end_symbol,
                                     torch.Tensor(Ass_res_his_current_round), task, opponent_model, prediction_horizon, loss_type)
            u_opt = u_opt.detach().numpy()
        print("Optimize time spent in the real world:" + str(T.time() - s1))

        # Conversion of continuous assigned amounts to discrete values
        decimal_array = u_opt * Agent_tobe_departed_number
        integer_array = np.floor(decimal_array).astype("int")
        res = max(0, Agent_tobe_departed_number - np.sum(
            integer_array).item())  # Number of agents remaining to be assigned
        while (res != 0):
            gaps = decimal_array - integer_array
            indices = np.argsort(gaps)
            integer_array[indices[-1]] += 1
            res -= 1

        print("Optimal Assignment in this batch:" + str(integer_array))
        action_sequence = np.vstack([action_sequence, np.concatenate([[adv_cur_time], integer_array])])

        # Updating Ass_res_his, Agent_tobe_ass, s1_len and s2_len
        Eff_actions_index = np.where(integer_array != 0)[0]
        Eff_actions_len = len(Eff_actions_index)
        # Updating Ass_res_his
        array1 = np.array([[i + 1, adv_cur_time + Time[i] * 2, adv_cur_time + Time[i],
                            integer_array[i] / Agent_number] for i in Eff_actions_index]).T
        Ass_res_his[:, s1_len:(s1_len + Eff_actions_len)] = array1
        # Updating Agent_tobe_ass
        array2 = np.array([[adv_cur_time + Time[i] * 2,
                            integer_array[i] / Agent_number] for i in Eff_actions_index]).T
        Agent_tobe_ass[:, s2_len:(s2_len + Eff_actions_len)] = array2
        # Updating s1_len and s2_len
        s1_len += Eff_actions_len
        s2_len += Eff_actions_len

        # Deleting tasks that were not identified in the previous round but have been implemented in this round
        s2_len -= 1
        Agent_tobe_ass = Agent_tobe_ass[:, 1:]

        # Recursing the environment to the next round start moment
        next_act_time = np.min(Agent_tobe_ass[0, :s2_len]).item()
        if Adv_phase < len(Adv_time):
            next_act_time = min(next_act_time, Adv_time[Adv_phase])
        cols = np.where((Ass_res_his[2, :] > adv_cur_time) & (Ass_res_his[2, :] <= next_act_time))[0]
        action_sequence_current_round = Ass_res_his[:, cols].copy()

        anti_capability_origin = anti_capability.clone()
        anti_capability, opponent_his_data = Env.Env_recursion(adv_cur_time, next_act_time,
                                                               action_sequence_current_round,
                                                               KO, anti_capability, opponent_his_data, args)

        print("Before weakening: " + str(anti_capability_origin))
        print("Simulation start time: " + str(adv_cur_time))
        print("Simulation end time: " + str(next_act_time))
        print("After weakening:" + str(anti_capability))
        print(" ")
        adv_cur_time = next_act_time

        # Model fine-tuning
        if (retrain and ((opponent_his_data[-1, 0] - train_time) >= 0.2)):
            train_time = opponent_his_data[-1, 0].copy()

            retrain_data = opponent_his_data.copy()
            retrain_label = torch.Tensor(retrain_data[:, 1:])
            for e in range(retrain_data.shape[0]):
                retrain_label[e, :] = Env.forward_equations(None, retrain_data[e, 1:], KE, KR, PNi, Bij, exp, beta, None)

            if deception:  # Existing deception of opponents
                retrain_data = torch.Tensor(retrain_data[:, 1:] * DLi)
                retrain_label = retrain_label * DLi
                train_loader = DataLoader(TensorDataset(retrain_data, retrain_label), batch_size=64, shuffle=True)
                trainable_layers = [['fc1'], ['fc3'], ['ae1'], ['fc1', 'fc3'], ['fc1', 'ae1'], ['fc3', 'ae1'],
                                    ['fc1', 'fc3', 'ae1'], ['all']]
            else:  # Without deception of opponents
                retrain_data1 = torch.Tensor(retrain_data[:, 1:])
                retrain_data2 = torch.zeros([retrain_data.shape[0], P, 3])
                for o in range(3):
                    retrain_data2[:, :, o] = retrain_data1 ** o
                retrain_label = retrain_label
                train_loader = DataLoader(TensorDataset(retrain_data1, retrain_data2, retrain_label), batch_size=64, shuffle=True)
                trainable_layers = [['fc1'], ['fc3'], ['fc1', 'fc3'], ['all']]

            # Fine-tuning in parallel
            parallel_fine_tune_num = len(trainable_layers)
            if not model_copies:
                model_copies = [copy.deepcopy(opponent_model) for _ in range(parallel_fine_tune_num)]
            args_list = [(train_loader, model_copies[i], 2000, trainable_layers[i]) for i in
                         range(parallel_fine_tune_num)]
            with Pool(processes=parallel_fine_tune_num) as pool:
                results = pool.starmap(NN_model.model_fine_tune, args_list)
            model_sequence, loss_sequence = list(zip(*results))

            # Selecting the model with highest fidelity as the current opponent model
            fidelities = [Model_fine_tuning.fidelity_calculation(train_loader, m) for m in model_sequence]
            max_fidelity_index = min(range(len(fidelities)), key=lambda i: fidelities[i])
            opponent_model, loss = model_sequence[max_fidelity_index], loss_sequence[max_fidelity_index]

            # Saving model and loss values
            # Saving model
            if not deception:
                torch.save(opponent_model[0].state_dict(),
                           model_path + str(Adv_phase-1) + "_retrain_number=" + str(retrain_number) + ".pkl")
            else:
                torch.save(opponent_model[0].state_dict(),
                           sys.path[0] + r"\trained_NN\ae1_Adv_phase=" + str(Adv_phase-1) + ".pkl")
                torch.save(opponent_model[1].state_dict(),
                           model_path + str(Adv_phase-1) + "_retrain_number=" + str(retrain_number) + ".pkl")
                torch.save(opponent_model[2].state_dict(),
                           sys.path[0] + r"\trained_NN\ae2_Adv_phase=" + str(Adv_phase-1) + ".pkl")
            # Saving loss values
            np.save(data_path + "loss_Adv_phase=" + str(Adv_phase-1) + "_retrain_number=" + str(retrain_number)
                    + "_KE=" + str(KE) + "_KR=" + str(KR) + "_KO=" + str(KO) + ".npy", np.array(loss))

            # Stop fine-tuning
            similarity = Model_fine_tuning.similarity_calculation(train_loader, target_opponent_model, opponent_model)
            print("Fidelity:"+str(similarity))
            if similarity > 0.95 or opponent_his_data[-1, 0] > (opponent_his_data[0, 0] + 1.0):
                retrain = False
                model = Model_fine_tuning.load_model(P, order, Adv_phase-1)
                encoder, decoder = Model_fine_tuning.load_encoder_decoder(P, Adv_phase-1)
                opponent_model = [model]
                if deception:
                    print("deception:" + str(deception))
                    opponent_model = [encoder, model, decoder]

                Model_fine_tuning.print_parameters_in_model(model)

            retrain_number += 1

    # Sample the adversarial process data at intervals of 0.05 seconds.
    opponent_his_data = Data_save.Sampling(opponent_his_data, given_intervals=0.05)
    # Calculate the exchange and recovery velocity of opponents.
    opponent_exc_vel_data, opponent_rec_vel_data = Data_save.Calculate_velocity(P, opponent_his_data, KE, KR, PNi, Bij, exp, beta)
    # Calculate the exchange and recovery amount of opponents.
    opponent_exc_data, opponent_rec_data = Data_save.Calculate_amount(opponent_exc_vel_data, opponent_rec_vel_data)
    # Save data as npy.
    Data_save.Save_data_as_npy(data_path, [Adv_phase, KE, KR, KO], opponent_his_data, opponent_exc_data, opponent_rec_data,
                           opponent_exc_vel_data, opponent_rec_vel_data, action_sequence, Ass_res_his)