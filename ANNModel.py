import torch
from torch import nn


class ANNModel(nn.Module):
    """
    三层简单的神经网络，激活函数采用sigmoid
    """

    def __init__(self, inputDimension: int, hiddenDimension: int, outputDimension: int):
        super(ANNModel, self).__init__()
        self.inputDimension = inputDimension
        self.hiddenDimension = hiddenDimension
        self.outputDimension = outputDimension
        self.linear_W1 = nn.Linear(inputDimension, hiddenDimension, bias=True, dtype=torch.float64)  # W1
        self.linear_W2 = nn.Linear(hiddenDimension, outputDimension, bias=True, dtype=torch.float64)  # W2
        self.b1 = 0.5
        self.b2 = 0.5
        self.activationFunc = nn.Sigmoid()

    def forward(self, x):
        A1 = self.linear_W1(x)+self.b1
        Z1 = self.activationFunc(A1)
        A2 = self.linear_W2(Z1)+self.b2
        Z2 = self.activationFunc(A2)
        return Z2
