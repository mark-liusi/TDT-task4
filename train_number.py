from torchvision import datasets
import torchvision.transforms as transforms
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 数据预处理：添加标准化
# 数据预处理：PNG 转 1×28×28，并做与 MNIST 一致的标准化
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 强制灰度，得到 1 通道
    transforms.Resize((28, 28)),                  # 统一为 28×28，匹配全连接输入
    transforms.ToTensor(),                        # 转张量，归一化到 [0,1]
    transforms.Normalize((0.1307,), (0.3081,))    # 先用 MNIST 均值方差，训练更稳
])

# 用 ImageFolder 读取“类名即标签”的层级目录（如 0、1、...、9）
full_data = datasets.ImageFolder(
    root="./number",      # ⚠️ 改成你的 PNG 根目录
    transform=transform
)
print("classes:", full_data.classes)  # 查看类名与标签映射是否符合预期

# 如果只有一个总文件夹，就按 8:2 划分为训练/测试
n_total = len(full_data)
n_train = int(0.8 * n_total)
n_test = n_total - n_train
train_data, test_data = torch.utils.data.random_split(
    full_data, [n_train, n_test],
    generator=torch.Generator().manual_seed(42)  # 固定随机种子
)

batch_size = 64
num_workers = 0

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size,
    shuffle=True, num_workers=num_workers        # 训练集打乱
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size,
    shuffle=False, num_workers=num_workers       # 测试集不打乱
)


# 改进的网络结构
class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.bn1 = nn.BatchNorm1d(512)  # 添加批标准化
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

model = ImprovedNet()
print("改进的模型结构：")
print(model)

# 使用更好的优化器和学习率调度
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)  # 学习率衰减

# 训练过程改进
n_epochs = 30  # 减少轮数，但提高质量
train_losses = []
train_accuracies = []

print("开始训练...")
for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 每100个batch打印一次进度
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}')
    
    # 计算平均损失和准确率
    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    
    # 更新学习率
    scheduler.step()
    
    print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}')
    
    # 每5轮在验证集上测试一次
    if (epoch + 1) % 5 == 0:
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        print(f'  -> 验证集: Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')

print("训练完成！")

# 最终测试
print("\n" + "="*50)
print("最终测试结果：")
print("="*50)

model.eval()
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
all_preds = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
        
        correct = pred.eq(target)
        for i in range(target.size(0)):
            label = target[i].item()
            class_correct[label] += correct[i].item()
            class_total[label] += 1

test_loss = test_loss / len(test_loader.dataset)
print(f'测试集损失: {test_loss:.6f}')

# 各类别准确率
print("\n各数字的识别准确率：")
for i in range(10):
    if class_total[i] > 0:
        accuracy = 100 * class_correct[i] / class_total[i]
        print(f'数字 {i}: {accuracy:.1f}% ({int(class_correct[i])}/{int(class_total[i])})')

# 总体准确率
overall_accuracy = 100. * sum(class_correct) / sum(class_total)
print(f'\n总体准确率: {overall_accuracy:.2f}% ({int(sum(class_correct))}/{int(sum(class_total))})')

# 保存改进的模型
torch.save(model.state_dict(), "mnist_fc_improved.pth")
print(f"\n改进的模型已保存到 mnist_fc_improved.pth")

# 显示训练曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('训练损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies)
plt.title('训练准确率')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

