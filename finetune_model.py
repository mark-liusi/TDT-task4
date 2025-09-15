# 终极解决方案：在你的数据上微调MNIST模型
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif']= ['Noto Sans CJK SC','Noto Sans CJK JP','WenQuanYi Micro Hei','SimHei']  # 支持中文的备选字体
matplotlib.rcParams['axes.unicode_minus']= False  # 允许坐标轴显示负号


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 使用效果较好的通用标准化
    ]
)

# 加载你的数据集
full_dataset = datasets.ImageFolder(root="./number", transform=transform)
print(f"总数据量: {len(full_dataset)}")
print(f"类别: {full_dataset.classes}")

# 分割数据集：80%训练，20%测试
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 模型定义（与原模型完全一致）
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


# 标签重映射函数
def remap_labels(labels):
    return torch.tensor([l + 1 for l in labels])


# 加载预训练模型
model = ImprovedNet().to(device)
model.load_state_dict(torch.load("mnist_fc_improved.pth", map_location=device))
print("✅ 成功加载MNIST预训练模型")

# 微调设置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 较小的学习率用于微调
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 训练函数
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = remap_labels(target).to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# 测试函数
def test_epoch(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = remap_labels(target).to(device)

            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# 微调训练
print("\n开始微调训练...")
num_epochs = 20  # 微调不需要太多轮
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    # 训练
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # 测试
    test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    # 学习率调度
    scheduler.step()

    print(
        f"Epoch {epoch+1:2d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:6.2f}%, "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:6.2f}%"
    )

# 保存微调后的模型
torch.save(model.state_dict(), "mnist_finetuned_on_custom.pth")
print(f"\n✅ 微调完成！模型已保存为 'mnist_finetuned_on_custom.pth'")
print(f"🎉 最终测试准确率: {test_accuracies[-1]:.2f}%")

# 绘制训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Test Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("finetuning_curves.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n📊 训练曲线已保存为 'finetuning_curves.png'")
print(f"💡 现在你可以用微调后的模型获得更好的识别效果！")
