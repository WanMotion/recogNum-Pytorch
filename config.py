from torch import nn

# 设定基本参数
inputDimension = 28 * 28  # 输入层
hiddenDimension = 100  # 隐藏层
outputDimension = 10  # 输出层
epoch = 3000
batchSize = 64
learningRate = 0.001
TEST_SIZE = 100  # 测试集大小
TRAIN_SIZE = 2000  # 训练集大小
loss = nn.MSELoss(size_average=False, reduce=True)  # 损失函数

# 输出位置
outputDir = "output"  # 训练结果输出位置
checkpointOutOutDir = "output/checkpoint"  # 检查点输出位置
checkpointPath = ""  # 为空时，表示不从断点继续训练

# 测试集所用参数位置
testAgsPthPath = "output/1630248264_epoch_3000.pth"
