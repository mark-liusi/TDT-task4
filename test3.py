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

# 改进的数据预处理：使用通用标准化而不是MNIST专用标准化
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),  # 确保是灰度图
        transforms.Resize((28, 28)),  # 调整所有图片到28x28
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 改进：使用通用标准化，适应不同数据分布
    ]
)

# 加载你的自定义数据集（使用ImageFolder）
test_data = datasets.ImageFolder(
    root="./number", transform=transform  # 使用你的number文件夹
)

print(f"发现的类别: {test_data.classes}")
print(f"类别映射: {test_data.class_to_idx}")
print(f"数据集大小: {len(test_data)}")


# 创建类别映射：将文件夹名（字符串）映射到实际数字
# 注意：ImageFolder将类别按字母顺序映射：'1'->0, '2'->1, '3'->2, '4'->3, '5'->4
# 但我们希望：'1'->1, '2'->2, '3'->3, '4'->4, '5'->5
def remap_labels(target):
    # ImageFolder的映射：{'1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
    # 我们需要的映射：1->1, 2->2, 3->3, 4->4, 5->5
    return target + 1  # 简单地加1


print(f"注意：类别映射调整后，文件夹'1'->标签1, '2'->标签2, 等等")

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
        # 重新映射标签
        target = torch.tensor([remap_labels(t.item()) for t in target])

        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)  # 累加测试总损失
        _, pred = torch.max(output, 1)  # 取每行最大logit的索引作为预测类别
        correct = pred.eq(target)  # 逐元素比较，得到布尔向量
        bs = target.size(0)  # 当前batch真实大小（最后一批可能小）
        for i in range(bs):
            label = int(target[i].item())
            if label < 10:  # 只统计0-9的类别
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
img = images[1]  # 第2张图
original_label = labels[1].item()  # ImageFolder的原始标签 (0-4)
true_label = remap_labels(original_label)  # 重新映射的标签 (1-5)

# 用加载的模型预测
with torch.no_grad():
    output = model(img.view(-1, 28 * 28))  # 单张图要展平成 [1, 784]
    pred = torch.argmax(output, dim=1).item()

print(f"真实标签: {true_label}, 预测结果: {pred}")
print(f"(原始ImageFolder标签: {original_label} -> 重映射为: {true_label})")

# 显示这张图片
plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"True Label: {true_label}, Predicted: {pred}")
plt.axis("off")  # 不显示坐标轴
plt.show()

print("自定义数据集推理完成！")
print(f"注意：这次使用的是你的自定义number数据集")
print(f"- 数据集包含类别：{test_data.classes}")
print(f"- 总共 {len(test_data)} 张图片")
print(f"- 图片已调整为28x28灰度图以匹配模型")
print(f"- 类别映射：文件夹'1'->标签1, '2'->2, '3'->3, '4'->4, '5'->5")
