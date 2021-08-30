import torch
import torchvision.transforms
from torch.utils.data import Dataset
import torchvision


class MnistDataSets(Dataset):
    """
    自定义数据集，获取mnist数据集后，正则化
    """

    def __init__(self, train: bool, size: int):
        super(MnistDataSets, self).__init__()
        if train:
            self.data = torchvision.datasets.MNIST(root="dataset", train=True,
                                                   transform=torchvision.transforms.ToTensor()
                                                   , download=True)
        else:
            self.data = torchvision.datasets.MNIST(root="dataset", train=False,
                                                   transform=torchvision.transforms.ToTensor()
                                                   , download=True)
        self.transData = self.data.data.clone()
        self.transData = self.transData.double().reshape(-1, 28 * 28)[0:size]
        self.transData = (self.transData - 127.5) / 127.5
        self.transLabel = torch.zeros(self.data.__len__(), 10, dtype=torch.float64)
        self.transLabel[range(self.data.__len__()), self.data.targets.data] = 1.0
        self.transLabel = self.transLabel[0:size]
        self.size = size

    def __getitem__(self, index):
        return self.transData[index], self.transLabel[index]

    def __len__(self):
        return self.size
