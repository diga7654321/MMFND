from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


'''专家网络'''
class Expert_net(nn.Module):
    def __init__(self, feature_dim, expert_dim):
        super(Expert_net, self).__init__()

        p = 0
        self.dnn_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, expert_dim),
            nn.ReLU(),
            nn.Dropout(p)
        )

    def forward(self, x):
        out = self.dnn_layer(x)
        return out


'''特征提取层'''
class Extraction_Network(nn.Module):
    '''FeatureDim-输入数据的维数; ExpertOutDim-每个Expert输出的维数; TaskExpertNum-任务特定专家数;
       CommonExpertNum-共享专家数; GateNum-gate数(2表示最后一层，3表示中间层)'''

    def __init__(self, FeatureDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, GateNum):
        super(Extraction_Network, self).__init__()

        self.GateNum = GateNum  # 输出几个Gate的结果，2表示最后一层只输出两个任务的Gate，3表示还要输出中间共享层的Gate

        '''两个任务模块，一个共享模块'''
        self.n_task = 2
        self.n_share = 1

        '''TaskA-Experts'''
        for i in range(TaskExpertNum):
            setattr(self, "expert_layer" + str(i + 1), Expert_net(FeatureDim, ExpertOutDim).cuda())
        self.Experts_A = [getattr(self, "expert_layer" + str(i + 1)) for i in
                          range(TaskExpertNum)]  # Experts_A模块，TaskExpertNum个Expert
        '''Shared-Experts'''
        for i in range(CommonExpertNum):
            setattr(self, "expert_layer" + str(i + 1), Expert_net(FeatureDim, ExpertOutDim).cuda())
        self.Experts_Shared = [getattr(self, "expert_layer" + str(i + 1)) for i in
                               range(CommonExpertNum)]  # Experts_Shared模块，CommonExpertNum个Expert
        '''TaskB-Experts'''
        for i in range(TaskExpertNum):
            setattr(self, "expert_layer" + str(i + 1), Expert_net(FeatureDim, ExpertOutDim).cuda())
        self.Experts_B = [getattr(self, "expert_layer" + str(i + 1)) for i in
                          range(TaskExpertNum)]  # Experts_B模块，TaskExpertNum个Expert

        '''Task_Gate网络结构'''
        for i in range(self.n_task):
            setattr(self, "gate_layer" + str(i + 1),
                    nn.Sequential(nn.Linear(FeatureDim, TaskExpertNum + CommonExpertNum),
                                  nn.Softmax(dim=1)).cuda())
        self.Task_Gates = [getattr(self, "gate_layer" + str(i + 1)) for i in
                           range(self.n_task)]  # 为每个gate创建一个lr+softmax

        '''Shared_Gate网络结构'''
        for i in range(self.n_share):
            setattr(self, "gate_layer" + str(i + 1),
                    nn.Sequential(nn.Linear(FeatureDim, 2 * TaskExpertNum + CommonExpertNum),
                                  nn.Softmax(dim=1)).cuda())
        self.Shared_Gates = [getattr(self, "gate_layer" + str(i + 1)) for i in range(self.n_share)]  # 共享gate

    def forward(self, x_A, x_S, x_B):
        '''Experts_A模块输出'''
        Experts_A_Out = [expert(x_A) for expert in self.Experts_A]  #
        Experts_A_Out = torch.cat(([expert[:, np.newaxis, :] for expert in Experts_A_Out]),
                                  dim=1)  # 维度 (bs,TaskExpertNum,ExpertOutDim)
        '''Experts_Shared模块输出'''
        Experts_Shared_Out = [expert(x_S) for expert in self.Experts_Shared]  #
        Experts_Shared_Out = torch.cat(([expert[:, np.newaxis, :] for expert in Experts_Shared_Out]),
                                       dim=1)  # 维度 (bs,CommonExpertNum,ExpertOutDim)
        '''Experts_B模块输出'''
        Experts_B_Out = [expert(x_B) for expert in self.Experts_B]  #
        Experts_B_Out = torch.cat(([expert[:, np.newaxis, :] for expert in Experts_B_Out]),
                                  dim=1)  # 维度 (bs,TaskExpertNum,ExpertOutDim)

        '''Gate_A的权重'''
        Gate_A = self.Task_Gates[0](x_A)  # 维度 n_task个(bs,TaskExpertNum+CommonExpertNum)
        '''Gate_Shared的权重'''
        if self.GateNum == 3:
            Gate_Shared = self.Shared_Gates[0](x_S)  # 维度 n_task个(bs,2*TaskExpertNum+CommonExpertNum)
        '''Gate_B的权重'''
        Gate_B = self.Task_Gates[1](x_B)  # 维度 n_task个(bs,TaskExpertNum+CommonExpertNum)

        '''GateA输出'''
        g = Gate_A.unsqueeze(2)  # 维度(bs,TaskExpertNum+CommonExpertNum,1)
        experts = torch.cat([Experts_A_Out, Experts_Shared_Out],
                            dim=1)  # 维度(bs,TaskExpertNum+CommonExpertNum,ExpertOutDim)
        Gate_A_Out = torch.matmul(experts.transpose(1, 2), g)  # 维度(bs,ExpertOutDim,1)
        Gate_A_Out = Gate_A_Out.squeeze(2)  # 维度(bs,ExpertOutDim)
        '''GateShared输出'''
        if self.GateNum == 3:
            g = Gate_Shared.unsqueeze(2)  # 维度(bs,2*TaskExpertNum+CommonExpertNum,1)
            experts = torch.cat([Experts_A_Out, Experts_Shared_Out, Experts_B_Out],
                                dim=1)  # 维度(bs,2*TaskExpertNum+CommonExpertNum,ExpertOutDim)
            Gate_Shared_Out = torch.matmul(experts.transpose(1, 2), g)  # 维度(bs,ExpertOutDim,1)
            Gate_Shared_Out = Gate_Shared_Out.squeeze(2)  # 维度(bs,ExpertOutDim)
        '''GateB输出'''
        g = Gate_B.unsqueeze(2)  # 维度(bs,TaskExpertNum+CommonExpertNum,1)
        experts = torch.cat([Experts_B_Out, Experts_Shared_Out],
                            dim=1)  # 维度(bs,TaskExpertNum+CommonExpertNum,ExpertOutDim)
        Gate_B_Out = torch.matmul(experts.transpose(1, 2), g)  # 维度(bs,ExpertOutDim,1)
        Gate_B_Out = Gate_B_Out.squeeze(2)  # 维度(bs,ExpertOutDim)

        if self.GateNum == 3:
            return Gate_A_Out, Gate_Shared_Out, Gate_B_Out
        else:
            return Gate_A_Out, Gate_B_Out


class PLE(nn.Module):
    # FeatureDim-输入数据的维数;ExpertOutDim-每个Expert输出的维数;TaskExpertNum-任务特定专家数;CommonExpertNum-共享专家数;n_task-任务数(gate数)
    def __init__(self, FeatureDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, n_task=2):
        super(PLE, self).__init__()
        # self.FeatureDim = x.shape[1]

        '''一层Extraction_Network，一层CGC'''
        self.Extraction_layer1 = Extraction_Network(FeatureDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, GateNum=3)
        self.CGC = Extraction_Network(ExpertOutDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, GateNum=2)

    def forward(self, x):
        Output_A, Output_Shared, Output_B = self.Extraction_layer1(x, x, x)
        Gate_A_Out, Gate_B_Out = self.CGC(Output_A, Output_Shared, Output_B)

        return Gate_A_Out, Gate_B_Out


# Model = PLE(FeatureDim=X_train.shape[1], ExpertOutDim=64, TaskExpertNum=1, CommonExpertNum=1).to(device)
# optimizer = torch.optim.Adam(Model.parameters(), lr=0.01)
# loss_func = nn.MSELoss().to(device)

# nParams = sum([p.nelement() for p in Model.parameters()])
# print('* number of parameters: %d' % nParams)


