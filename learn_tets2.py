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

# 改进的数据预处理：添加标准化
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,)
        ),  # 改进1：MNIST数据集的均值和标准差标准化，提高训练稳定性
    ]
)

# 只加载测试集（用于验证模型效果）
test_data = datasets.MNIST(
    root="./mnist_data", train=False, download=True, transform=transform  # 测试集
)

batch_size = 64  # 改进2：增大batch size从20到64，提高训练效率和稳定性
num_work = 0  # 你自定义的变量名；传给DataLoader时要用num_workers

# 只需要测试集的DataLoader
test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_work,  # 测试集不打乱
)


# 改进的网络结构（必须与训练时完全一致）
class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.bn1 = nn.BatchNorm1d(512)  # 改进3：添加批标准化层，防止梯度消失，加速收敛
        self.fc2 = nn.Linear(512, 256)  # 改进4：使用递减的神经元数量(512->256->128)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 10)  # 改进5：增加了一层，让网络更深，学习能力更强
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(
            self.bn1(self.fc1(x))
        )  # 改进6：每层都使用 线性->批标准化->激活函数 的顺序
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)  # 最后一层不使用激活函数，交叉熵损失会处理
        return x


# ===== 加载已保存的改进模型 =====
print("正在加载改进的已保存模型...")
model = ImprovedNet()  # 必须用相同结构
model.load_state_dict(
    torch.load("mnist_fc_improved.pth")
)  # 改进7：加载用Adam优化器和学习率调度训练的改进模型
model.eval()  # 设置为评估模式（关闭Dropout和BatchNorm的训练模式）
print("改进模型加载成功！")

# 损失函数（用于测试评估）
criterion = nn.CrossEntropyLoss()
# ===== 在测试集上评估模型性能 =====
print("正在评估模型性能...")
test_loss = 0.0
class_correct = list(0.0 for i in range(10))  # 各类预测正确数
class_total = list(0.0 for i in range(10))  # 各类样本总数
classes = [str(i) for i in range(10)]

with torch.no_grad():  # 不求梯度，省内存
    for data, target in test_loader:
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)  # 累加测试总损失
        _, pred = torch.max(output, 1)  # 取每行最大logit的索引作为预测类别
        correct = pred.eq(target)  # 逐元素比较，得到布尔向量
        bs = target.size(0)  # 当前batch真实大小（最后一批可能小）
        for i in range(bs):
            label = int(target[i].item())
            class_correct[label] += int(correct[i].item())  # 注意这里要 item()
            class_total[label] += 1

test_loss = test_loss / len(test_loader.dataset)  # 平均测试损失
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

# ===== 单张图片预测示例 =====

# 从测试集取一张图片进行预测示例
print("\n===== 单张图片预测示例 =====")
data_iter = iter(test_loader)
images, labels = next(data_iter)
img = images[1]  # 第1张图
label = labels[1].item()

# 用加载的模型预测
with torch.no_grad():
    output = model(img.view(-1, 28 * 28))  # 单张图要展平成 [1, 784]
    pred = torch.argmax(output, dim=1).item()

print(f"真实标签: {label}, 预测结果: {pred}")

# 显示这张图片
plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"True Label: {label}, Predicted: {pred}")
plt.axis("off")  # 不显示坐标轴
plt.show()

# print("改进模型推理完成！")
# print(f"注意：改进模型使用了以下7个关键改进：")
# print(f"1. 数据标准化：使用MNIST标准均值和方差")
# print(f"2. 增大batch_size：从20提升到64")
# print(f"3. 批标准化：每层添加BatchNorm防止梯度消失")
# print(f"4. 递减网络结构：512->256->128->10")
# print(f"5. 增加网络深度：从3层增加到4层")
# print(f"6. 改进激活顺序：线性->BN->ReLU->Dropout")
# print(f"7. Adam优化器+学习率调度：替代SGD，训练更稳定")
# print(f"最终准确率从8%提升到了98.67%！")
