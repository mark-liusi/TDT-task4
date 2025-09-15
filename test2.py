# import pandas as pd          # 导入 pandas，用于读取和处理表格数据
# import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot，用于绘图
# import numpy as np           # 导入 numpy，用于数值计算

# train= pd.read_csv("/home/liusi/下载/mnist_data/mnist_train.csv")  
# # 从指定路径读取 mnist_train.csv，第一列是标签，后面 784 列是 28x28 的像素

# def show_grid(df, n=25):     # 定义函数 show_grid，df 为数据表，n 为展示的样本数（默认 25）
#     n= min(n, len(df))       # 取 n 和数据行数的较小值，防止越界
#     cols= int(np.ceil(np.sqrt(n)))   # 列数取 sqrt(n) 的上取整，使布局尽量接近正方形
#     rows= int(np.ceil(n/cols))       # 根据列数算出需要的行数，上取整保证能放下所有图像
#     fig= plt.figure(figsize=(cols*2, rows*2))  # 新建画布，大小随行列数调整，每个小图大约 2x2 英寸

#     for i in range(n):       # 循环遍历前 n 行样本
#         ax= fig.add_subplot(rows, cols, i+1)  # 在 rows×cols 网格中创建第 i+1 个子图
#         label= int(df.iloc[i, 0])   # 取第 i 行第 0 列（标签列）的值，转成整数
#         img= df.iloc[i, 1:].values.reshape(28, 28)  
#         # 取第 i 行从第 1 列到最后一列（像素数据），转成 numpy 数组并 reshape 为 28×28 矩阵

#         ax.imshow(img, cmap="gray")   # 显示图像，灰度色图
#         ax.set_title(str(label))      # 子图标题设置为样本的标签
#         ax.axis("off")                # 去掉坐标轴

#     plt.tight_layout()   # 自动调整子图间距，避免标题和图像重叠
#     plt.show()           # 显示整张图

# show_grid(train, n=100)   # 调用函数，展示前 100 个样本

from torchvision import datasets         # 导入 torchvision 的数据集模块
import torchvision.transforms as transforms  # 导入图像预处理模块
import torch                             # 导入 PyTorch
import numpy as np                       # 导入 NumPy
import torch.nn as nn                    # 导入神经网络模块
import torch.nn.functional as F          # 导入常用激活函数和功能函数

# 非并行加载就填0
num_workers=0                            # 数据加载时的工作进程数（0 表示不并行）
batch_size=20                            # 每个 batch 的大小

# 转换成张量
transform=transforms.ToTensor()          # 把图片转为 PyTorch 张量，像素值缩放到 [0,1]

# 下载训练集（若本地已有将直接使用缓存）
train_data=datasets.MNIST(
    root="./mnist_data",                 # 数据存放路径
    train=True,                          # 是否是训练集
    download=True,                       # 若没有则下载
    transform=transform                  # 应用的预处理
)

# 下载测试集
test_data=datasets.MNIST(
    root="./mnist_data",
    train=False,                         # 测试集
    download=True,
    transform=transform
)

# 创建数据加载器
train_loader=torch.utils.data.DataLoader(
    train_data, batch_size=batch_size,
    shuffle=True, num_workers=num_workers  # 训练时打乱顺序
)
test_loader=torch.utils.data.DataLoader(
    test_data, batch_size=batch_size,
    shuffle=False, num_workers=num_workers # 测试时保持顺序
)

# 定义神经网络
class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()            # 调用父类构造函数
    hidden_1=512                          # 第一个隐藏层神经元数
    hidden_2=512                          # 第二个隐藏层神经元数
    self.fc1=nn.Linear(28*28,hidden_1)    # 全连接层1: 输入 784 -> 512
    self.fc2=nn.Linear(hidden_1,hidden_2) # 全连接层2: 512 -> 512
    self.fc3=nn.Linear(hidden_2,10)       # 输出层: 512 -> 10（10 类数字）
    self.dropout=nn.Dropout(0.2)          # Dropout，丢弃 20%，防止过拟合

  def forward(self,x):
    x=x.view(-1,28*28)                    # 将图像展平为 [batch,784]
    x=F.relu(self.fc1(x))                 # 第1层 + ReLU 激活
    x=self.dropout(x)                     # Dropout
    x=F.relu(self.fc2(x))                 # 第2层 + ReLU 激活
    x=self.dropout(x)                     # Dropout
    x=self.fc3(x)                         # 输出层（未加 softmax，因为交叉熵里会处理）
    return x

model=Net()                              # 实例化网络
print(model)                             # 打印网络结构

# 定义损失函数和优化器
criterion=nn.CrossEntropyLoss()          # 交叉熵损失（自动包含 softmax）
optimizer=torch.optim.SGD(               # 随机梯度下降优化器
    params=model.parameters(), lr=0.01
)

# 训练
n_epochs=50                              # 迭代次数
for epoch in range(n_epochs):
  model.train()                          # 设置为训练模式
  train_loss=0.0                         # 累计训练损失
  for data,target in train_loader:       # 遍历训练集
    optimizer.zero_grad()                # 梯度清零
    output=model(data)                   # 前向传播得到预测值 
    loss=criterion(output,target)        # 计算损失
    loss.backward()                      # 反向传播
    optimizer.step()                     # 更新参数
    train_loss+=loss.item()*data.size(0) # 累加损失
  train_loss=train_loss/len(train_loader.dataset)  # 平均损失
  print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1,train_loss))

# 测试
test_loss=0.0
class_correct=list(0. for i in range(10))   # 各类别正确预测数
class_total=list(0. for i in range(10))     # 各类别总数
classes=[str(i) for i in range(10)]         # 类别标签（字符串）

model.eval()                             # 设置为评估模式
with torch.no_grad():                    # 不计算梯度，节省内存
  for data,target in test_loader:        # 遍历测试集
    output=model(data)                   # 前向传播
    loss=criterion(output,target)        # 计算损失
    test_loss+=loss.item()*data.size(0)  # 累加损失
    _,pred=torch.max(output,1)           # 取概率最大的位置作为预测类别
    correct=pred.eq(target)              # 判断预测是否正确
    bs=target.size(0)                    # batch size
    for i in range(bs):                  # 遍历 batch 内每个样本
      label=int(target[i].item())        # 真实标签
      class_correct[label]+=int(correct[i].item()) # 累加正确数
      class_total[label]+=1              # 累加样本总数

test_loss=test_loss/len(test_loader.dataset)  # 平均测试损失
print('Test Loss: {:.6f}\n'.format(test_loss))

# 输出每一类的准确率
for i in range(10):
  if class_total[i]>0:
    print('Test Accuracy of %5s: %2d%% (%2d/%2d)'%(
      str(i),
      100*class_correct[i]/class_total[i],  # 准确率
      np.sum(class_correct[i]),             # 正确数
      np.sum(class_total[i])                # 总数
    ))
  else:
    print('Test Accuracy of %5s: N/A (no training examples)'%(classes[i]))

# 输出总体准确率
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)'%(
  100.*np.sum(class_correct)/np.sum(class_total),
  np.sum(class_correct),np.sum(class_total)
))


