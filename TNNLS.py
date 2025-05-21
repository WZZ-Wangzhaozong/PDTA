import main
# import NN.NN_model as NN_model
# import torch
# import sys
# import parameter
# class Model_fine_tuning:
#     @staticmethod
#     def print_parameters_in_model(model):
#         print("Parameters in IE network:")
#         for name, param in model.Son_swapnet.fc1.named_parameters():
#             print(param.data)
#
#         print("Parameters in BC network:")
#         for name, param in model.Son_swapnet.fc2.named_parameters():
#             print(param.data)
#
#         print("Parameters in TD network:")
#         for name, param in model.Son_swapnet.fc3.named_parameters():
#             print(param.data)
#
#         print("Parameters in sub-recovery network:")
#         for name, param in model.Son_recovernet.fc.named_parameters():
#             print(param.data)
#         print(" ")
#
# for Adv_phase in range(7):
#     print(Adv_phase)
#     print(sys.path[0] + r"\trained_NN\tnnls_based_Adv_phase=" + str(Adv_phase) + ".pkl")
#     print(parameter.P[Adv_phase])
#     model = NN_model.Opponent_model_explicit(channels=parameter.P[Adv_phase], order=3)
#     model.load_state_dict(torch.load(sys.path[0] + r"\trained_NN\tnnls_based_Adv_phase=" + str(Adv_phase) + ".pkl"))  # 执行方对任务方的建模
#     Model_fine_tuning.print_parameters_in_model(model)

strategy = "TNNLS"  # "Quantity-based"  # "AST", "TASE", "prediction-driven", "prediction-NN-driven"
main.main(strategy)