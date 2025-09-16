#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调用训练模型测试number文件夹中的图片
使用微调后的模型进行数字识别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random


# 定义改进的网络架构（与训练时保持一致）
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
        self.dropout = nn.Dropout(0.3)

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


def load_model(model_path):
    """加载训练好的模型"""
    print(f"正在加载模型: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型实例
    model = ImprovedNet()

    # 加载模型权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f"✅ 模型加载成功！使用设备: {device}")
        return model, device
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        return None, None


def get_data_loader(data_path, batch_size=32):
    """创建数据加载器"""
    # 数据预处理（与微调时保持一致）
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
            transforms.Resize((28, 28)),  # 调整大小到28x28
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5,), (0.5,)),  # 归一化到[-1, 1]
        ]
    )

    if not os.path.exists(data_path):
        print(f"❌ 数据路径不存在: {data_path}")
        return None

    try:
        # 使用ImageFolder加载数据
        dataset = datasets.ImageFolder(root=data_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        print(f"📁 数据集信息:")
        print(f"   路径: {data_path}")
        print(f"   类别: {dataset.classes}")
        print(f"   样本总数: {len(dataset)}")
        print(f"   类别映射: {dataset.class_to_idx}")

        return dataloader, dataset
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None, None


def test_model(model, dataloader, device, dataset):
    """测试模型并输出详细结果"""
    print("\n🧪 开始测试模型...")

    model.eval()
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}
    predictions = []
    true_labels = []
    confidences = []

    # 初始化类别统计
    for i, class_name in enumerate(dataset.classes):
        class_correct[i] = 0
        class_total[i] = 0

    # 类别映射：ImageFolder的0-4映射到模型的1-5
    def map_target_to_model(target):
        # ImageFolder: 0,1,2,3,4 对应文件夹 1,2,3,4,5
        # 模型输出: 0-9，其中1,2,3,4,5对应数字1,2,3,4,5
        return target + 1

    def map_model_to_target(model_pred):
        # 模型输出1,2,3,4,5映射回ImageFolder的0,1,2,3,4
        if 1 <= model_pred <= 5:
            return model_pred - 1
        else:
            return -1  # 无效预测

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            # 前向传播
            output = model(data)
            probabilities = F.softmax(output, dim=1)

            # 获取预测结果
            _, predicted = torch.max(output, 1)

            # 统计各类别准确率
            for i in range(target.size(0)):
                true_label = target[i].item()  # ImageFolder的标签 (0-4)
                model_pred = predicted[i].item()  # 模型预测 (0-9)
                mapped_pred = map_model_to_target(model_pred)  # 映射回 (0-4)

                # 获取置信度
                confidence = probabilities[i][model_pred].item()

                class_total[true_label] += 1
                total += 1

                if mapped_pred == true_label:
                    class_correct[true_label] += 1
                    correct += 1

                # 保存详细信息
                predictions.append(mapped_pred)
                true_labels.append(true_label)
                confidences.append(confidence)

            # 显示进度
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"   已处理 {(batch_idx + 1) * dataloader.batch_size} / {len(dataset)} 样本"
                )

    # 输出结果
    print(f"\n📊 测试结果:")
    print(f"   总样本数: {total}")
    print(f"   总体准确率: {100 * correct / total:.2f}% ({correct}/{total})")
    print(f"   平均置信度: {np.mean(confidences):.3f}")

    print(f"\n🔍 各类别详细结果:")
    for i, class_name in enumerate(dataset.classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(
                f"   数字 {class_name}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})"
            )
        else:
            print(f"   数字 {class_name}: 无测试样本")

    return predictions, true_labels, confidences


def visualize_sample_predictions(model, dataset, device, num_samples=12):
    """可视化一些样本的预测结果"""
    print(f"\n🖼️  随机展示 {num_samples} 个预测样本...")

    # 随机选择样本
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()

    # 类别映射函数
    def map_model_to_target(model_pred):
        if 1 <= model_pred <= 5:
            return model_pred - 1
        else:
            return -1

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            if i >= 12:  # 最多显示12个
                break

            # 获取样本
            image, true_label = dataset[idx]
            image_batch = image.unsqueeze(0).to(device)

            # 预测
            output = model(image_batch)
            probabilities = F.softmax(output, dim=1)
            confidence, model_predicted = torch.max(probabilities, 1)

            # 映射预测结果
            mapped_pred = map_model_to_target(model_predicted.item())

            # 显示图像
            img_np = image.squeeze().cpu().numpy()
            axes[i].imshow(img_np, cmap="gray")

            if mapped_pred >= 0 and mapped_pred < len(dataset.classes):
                pred_class = dataset.classes[mapped_pred]
            else:
                pred_class = f"无效({model_predicted.item()})"

            axes[i].set_title(
                f"真实: {dataset.classes[true_label]}\n"
                f"预测: {pred_class}\n"
                f"置信度: {confidence.item():.3f}",
                fontsize=10,
            )
            axes[i].axis("off")

            # 如果预测错误，用红色标题
            if mapped_pred != true_label:
                axes[i].set_title(
                    f"真实: {dataset.classes[true_label]}\n"
                    f"预测: {pred_class}\n"
                    f"置信度: {confidence.item():.3f}",
                    fontsize=10,
                    color="red",
                )

    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("💾 样本预测图已保存为 sample_predictions.png")


def test_single_image(model, device, image_path, dataset_classes):
    """测试单张图片"""
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        return

    # 数据预处理
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    def map_model_to_target(model_pred):
        if 1 <= model_pred <= 5:
            return model_pred - 1
        else:
            return -1

    try:
        # 加载和预处理图片
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(device)

        # 预测
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, model_predicted = torch.max(probabilities, 1)

        mapped_pred = map_model_to_target(model_predicted.item())

        print(f"📸 单张图片测试: {image_path}")
        if mapped_pred >= 0 and mapped_pred < len(dataset_classes):
            print(f"   预测结果: 数字 {dataset_classes[mapped_pred]}")
        else:
            print(f"   预测结果: 无效预测 (模型输出: {model_predicted.item()})")
        print(f"   置信度: {confidence.item():.3f}")

        # 显示所有数字类别的概率
        print("   各数字类别概率:")
        for digit in [1, 2, 3, 4, 5]:  # 只显示我们关心的数字
            prob = probabilities[0][digit].item()
            print(f"     数字 {digit}: {prob:.3f}")

    except Exception as e:
        print(f"❌ 单张图片测试失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 数字识别模型测试程序")
    print("=" * 60)

    # 配置参数
    model_path = "mnist_finetuned_on_custom.pth"  # 微调后的模型
    data_path = "./number"  # 测试数据路径
    batch_size = 32

    # 检查是否有微调模型，如果没有就使用原始模型
    if not os.path.exists(model_path):
        print(f"⚠️  微调模型不存在，尝试使用原始模型...")
        model_path = "mnist_fc_improved.pth"

    # 加载模型
    model, device = load_model(model_path)
    if model is None:
        return

    # 加载测试数据
    result = get_data_loader(data_path, batch_size)
    if result is None:
        return
    dataloader, dataset = result

    # 测试模型
    predictions, true_labels, confidences = test_model(
        model, dataloader, device, dataset
    )

    # 可视化样本预测
    visualize_sample_predictions(model, dataset, device)

    # 测试单张图片示例（如果存在的话）
    print(f"\n🔍 寻找单张图片进行测试...")
    for class_name in dataset.classes:
        class_dir = os.path.join(data_path, class_name)
        if os.path.exists(class_dir):
            images = [
                f
                for f in os.listdir(class_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if images:
                sample_image = os.path.join(class_dir, images[0])
                test_single_image(model, device, sample_image, dataset.classes)
                break

    print(f"\n✅ 测试完成！")


if __name__ == "__main__":
    main()
