import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.transform import Rotation as R
from math import cos as c
from math import sin as s


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.tanh(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.tanh(self.fc3(x))

        return F.tanh(x)


def Gen_data():
    _tmp = 0
    Angle_fixed = F.normalize(torch.rand(1, 3), dim=1)
    rand_ave = torch.tensor([0.5, 0.5, 0.5])
    RPY = (torch.rand(3) - rand_ave) * math.pi * 2

    R = RPY[0]
    P = RPY[1]
    Y = RPY[2]

    CR = c(R)
    CP = c(P)
    CY = c(Y)
    SR = s(R)
    SP = s(P)
    SY = s(Y)

    RotMat = torch.transpose(torch.tensor([[CY * CP, -SY * CR + CY * SP * SR, SY * SR + CY * SP * CR],
                                           [SY * CP, CY * CR + SY * SP * SR, -CY * SR + SY * SP * CR],
                                           [-SP, CP * SR, CP * CR]]), 0, 1)

    '''
    RotMat = R.from_euler('zyx', [RPY], degrees=False).as_matrix()
    RotMat = torch.tensor(RotMat)
    RotMat.view(3,3)
    '''
    # print(RotMat)
    Angle_moving = torch.matmul(RotMat, torch.transpose(Angle_fixed, 0, 1))

    # print(Angle_fixed, Angle_moving)
    '''
    for _index in range(3):
        _tmp[_index][0]=[Angle_fixed[_index],[Angle_moving[_index]]]
    '''
    # print(Angle_fixed,'\n',Angle_moving)#,'\n',_tmp)
    _data = torch.tensor([Angle_fixed[0][0], Angle_moving[0][0],
                          Angle_fixed[0][1], Angle_moving[1][0],
                          Angle_fixed[0][2], Angle_moving[2][0]])
    return _data, RPY


_xdata_list = []
_ydata_list = []

model = NeuralNetwork()
model.train()
train_loss = []
train_accu = []

i = 0

device = torch.device('cpu')
net = NeuralNetwork().to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.01)
for _ in range(15):
    _xdata_list=[]
    _ydata_list=[]
    for _ in range(100):
        _xdata, _ydata = Gen_data()
        _xdata_list.append(_xdata)
        _ydata_list.append(_ydata)

    x_data = torch.stack(_xdata_list)
    y_data = torch.stack(_ydata_list)
    for i in range(10000):
        net.train()
        x_data, y_data = x_data.to(device), y_data.to(device)
        optimizer.zero_grad()
        output = net(x_data)
        print(output.shape, y_data.shape)
        loss=loss_fn(output, y_data)
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(loss.item())

print(x_data.shape, y_data.shape)
# print(torch.transpose(Angle_fixed,0,1).shape,Angle_moving.shape)
# print(data, '\n', RPY)