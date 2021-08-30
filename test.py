import torch
from torch.utils.data import DataLoader

from config import *
from ANNModel import *
from dataSets import *
import os


def test():
    if testAgsPthPath == "" or not os.path.exists(testAgsPthPath):
        print("Error : 路径错误")
        return

    model = ANNModel(inputDimension, hiddenDimension, outputDimension)
    model.load_state_dict(torch.load(testAgsPthPath))

    # 准备测试数据
    testData = MnistDataSets(False, TEST_SIZE)
    testDataLoader = DataLoader(testData, batch_size=batchSize, shuffle=False)
    model.eval()
    totalCount=0
    for testX, testY in testDataLoader:
        valueY = model(testX)
        valueY.argmax(dim=1)
        count = torch.count_nonzero(valueY.argmax(dim=1) == testY.argmax(dim=1))
        totalCount+=count
    print(f"result:{totalCount/TEST_SIZE*100}%")

if __name__ == '__main__':
    test()

