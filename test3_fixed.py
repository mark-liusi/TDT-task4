# ===== 直接运行已保存的模型（改进版本） =====
from torchvision import datasets
import torchvision.transforms as transforms
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重现
torch.manual_seed(42)

# 数据预处理：使用MNIST标准化参数
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # 确保是灰度图
        transforms.Resize((28, 28)),  # 调整所有图片到28x28
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # 使用MNIST标准化参数
    ]
)

# 加载你的自定义数据集（使用ImageFolder）
test_data = datasets.ImageFolder(
    root="./number", transform=transform  # 使用你的number文件夹
)

print(f"发现的类别: {test_data.classes}")
print(f"类别映射: {test_data.class_to_idx}")
print(f"数据集大小: {len(test_data)}")
print(f"注意：ImageFolder自动映射：'1'->0, '2'->1, '3'->2, '4'->3, '5'->4")

batch_size = 64
num_work = 0

# 只需要测试集的DataLoader
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_work,
)


# 改进的网络结构（必须与训练时完全一致）
class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


# ===== 加载已保存的改进模型 =====
print("正在加载改进的已保存模型...")
model = ImprovedNet()
model.load_state_dict(torch.load("mnist_fc_improved.pth"))
model.eval()
print("改进模型加载成功！")

# 损失函数
criterion = nn.CrossEntropyLoss()

# ===== 在测试集上评估模型性能 =====
print("正在评估模型性能...")
test_loss = 0.0
class_correct = list(0.0 for i in range(10))
class_total = list(0.0 for i in range(10))

with torch.no_grad():
    for data, target in test_loader:
        # 不需要重新映射标签，直接使用ImageFolder的原始映射
        # target保持原样：'1'->0, '2'->1, '3'->2, '4'->3, '5'->4

        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct = pred.eq(target)
        bs = target.size(0)
        for i in range(bs):
            label = int(target[i].item())
            if label < 10:  # 只统计0-9的类别
                class_correct[label] += int(correct[i].item())
                class_total[label] += 1

test_loss = test_loss / len(test_loader.dataset)
print("Test Loss: {:.6f}\n".format(test_loss))

# 各类准确率
for i in range(10):
    if class_total[i] > 0:
        print(
            "Test Accuracy of %s: %2d%% (%2d/%2d)"
            % (
                str(i),
                100 * class_correct[i] / class_total[i],
                int(np.sum(class_correct[i])),
                int(np.sum(class_total[i])),
            )
        )
    else:
        print("Test Accuracy of %s: N/A (no samples)" % (str(i)))

# 总体准确率
overall_correct = int(np.sum(class_correct))
overall_total = int(np.sum(class_total))
print(
    "\nTest Accuracy (Overall): %2d%% (%2d/%2d)"
    % (100.0 * overall_correct / overall_total, overall_correct, overall_total)
)

print("\n=== 标签解释 ===")
print("标签0对应数字1")
print("标签1对应数字2")
print("标签2对应数字3")
print("标签3对应数字4")
print("标签4对应数字5")

print("自定义数据集推理完成！")
print(f"使用MNIST标准化参数 (0.1307, 0.3081)")
print(f"修正了标签映射问题")
