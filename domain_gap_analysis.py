#!/usr/bin/env python3
"""
域差距分析：对比原始模型在MNIST vs 自定义数据上的表现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# 模型定义
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


def test_model_on_dataset(model, dataloader, dataset_name, device, remap_labels=False):
    """测试模型在特定数据集上的表现"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            # 如果是自定义数据集，需要重新映射标签
            if remap_labels:
                target = target + 1  # ImageFolder的0-4映射到模型的1-5

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / total
    print(f"{dataset_name} 准确率: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


def main():
    print("=" * 60)
    print("🔬 域差距分析：原始模型在不同数据集上的表现")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载原始模型
    print("📥 加载原始MNIST模型...")
    model = ImprovedNet()
    model.load_state_dict(torch.load("mnist_fc_improved.pth", map_location=device))
    model.to(device)
    print("✅ 原始模型加载成功")

    # 1. 测试原始MNIST数据集上的表现
    print("\n🧪 测试1: 原始模型在MNIST测试集上的表现")
    mnist_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST专用标准化
        ]
    )

    mnist_test = datasets.MNIST(
        root="./mnist_data", train=False, download=False, transform=mnist_transform
    )
    mnist_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

    mnist_acc = test_model_on_dataset(
        model, mnist_loader, "MNIST", device, remap_labels=False
    )

    # 2. 测试自定义数据集上的表现（使用MNIST标准化）
    print("\n🧪 测试2: 原始模型在自定义数据上的表现（MNIST标准化）")
    custom_transform_mnist = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # 错误：用MNIST标准化
        ]
    )

    custom_dataset_mnist = datasets.ImageFolder(
        root="./number", transform=custom_transform_mnist
    )
    custom_loader_mnist = DataLoader(custom_dataset_mnist, batch_size=64, shuffle=False)

    custom_acc_mnist = test_model_on_dataset(
        model, custom_loader_mnist, "自定义数据(MNIST标准化)", device, remap_labels=True
    )

    # 3. 测试自定义数据集上的表现（使用通用标准化）
    print("\n🧪 测试3: 原始模型在自定义数据上的表现（通用标准化）")
    custom_transform_general = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # 改进：通用标准化
        ]
    )

    custom_dataset_general = datasets.ImageFolder(
        root="./number", transform=custom_transform_general
    )
    custom_loader_general = DataLoader(
        custom_dataset_general, batch_size=64, shuffle=False
    )

    custom_acc_general = test_model_on_dataset(
        model,
        custom_loader_general,
        "自定义数据(通用标准化)",
        device,
        remap_labels=True,
    )

    # 4. 对比微调模型的表现
    print("\n🧪 测试4: 微调模型在自定义数据上的表现")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finetuned_model = ImprovedNet()
    finetuned_model.load_state_dict(
        torch.load("mnist_finetuned_on_custom.pth", map_location=device)
    )
    finetuned_model.to(device)

    finetuned_acc = test_model_on_dataset(
        finetuned_model,
        custom_loader_general,
        "微调模型(自定义数据)",
        device,
        remap_labels=True,
    )

    # 结果总结
    print("\n" + "=" * 60)
    print("📊 结果总结")
    print("=" * 60)
    print(f"1. 原始模型 + MNIST数据集:        {mnist_acc:.2f}%")
    print(f"2. 原始模型 + 自定义数据(MNIST标准化): {custom_acc_mnist:.2f}%")
    print(f"3. 原始模型 + 自定义数据(通用标准化): {custom_acc_general:.2f}%")
    print(f"4. 微调模型 + 自定义数据:        {finetuned_acc:.2f}%")

    print(f"\n🔍 关键发现:")
    print(f"   📉 域差距损失: {mnist_acc - custom_acc_general:.2f}% (原始→自定义)")
    print(f"   📈 微调提升: {finetuned_acc - custom_acc_general:.2f}% (原始→微调)")
    print(
        f"   🎯 标准化影响: {custom_acc_general - custom_acc_mnist:.2f}% (MNIST标准化→通用标准化)"
    )

    # 可视化对比
    categories = [
        "MNIST\n(原始)",
        "自定义\n(MNIST标准化)",
        "自定义\n(通用标准化)",
        "自定义\n(微调模型)",
    ]
    accuracies = [mnist_acc, custom_acc_mnist, custom_acc_general, finetuned_acc]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, accuracies, color=["blue", "orange", "green", "red"])
    plt.ylabel("准确率 (%)")
    plt.title("模型在不同数据集上的表现对比")
    plt.ylim(0, 100)

    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("domain_gap_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\n💾 分析图表已保存为 'domain_gap_analysis.png'")


if __name__ == "__main__":
    main()
