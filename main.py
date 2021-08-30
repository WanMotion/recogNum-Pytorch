import os
import time

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from ANNModel import *
from dataSets import *
from config import *


def train():
    annModel=ANNModel(inputDimension, hiddenDimension, outputDimension)
    optimizer = torch.optim.SGD(annModel.parameters(), lr=learningRate)
    nowEpoch=0
    # 断点检查
    if checkpointPath!="" and os.path.exists(checkpointPath):
        check=torch.load(checkpointPath)
        annModel.load_state_dict(check['net'])
        optimizer.load_state_dict(check['optimizer'])
        nowEpoch=check['epoch']

    # 准备数据集
    trainData = MnistDataSets(True, TRAIN_SIZE)

    # 训练
    writer = SummaryWriter("logs")
    trainDataLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True)
    annModel.train()
    for i in range(nowEpoch,epoch):
        totalLoss = 0
        for trainX, trainY in trainDataLoader:
            output = annModel(trainX)
            optimizer.zero_grad()
            l = loss(output, trainY)
            l.backward()
            totalLoss += l.item()
            optimizer.step()
        writer.add_scalar("Loss", totalLoss, i)
        if (i + 1) % 100 == 0:
            print(f"epoch:{i},loss:{totalLoss}")
        if i + 1 >= 1000 and (i + 1) % 1000 == 0:
            checkpoint = {
                "net": annModel.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": i
            }
            if not os.path.exists(outputDir):
                os.mkdir(outputDir)
            if not os.path.exists(checkpointOutOutDir):
                os.mkdir(checkpointOutOutDir)
            torch.save(checkpoint, checkpointOutOutDir+"/"+str(int(time.time()))+"_epoch_"+str(epoch)+".pth")
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    torch.save(annModel.state_dict(),outputDir+"/"+str(int(time.time()))+"_epoch_"+str(epoch)+".pth")
    writer.close()

if __name__ == '__main__':
    train()
