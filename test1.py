import numpy as np
import cv2
# import torch
import matplotlib.pyplot as plt

# 1. 构造数据 (y = 2x + 3 + 噪声)
np.random.seed(0)
X = np.linspace(0, 10, 50)        # 输入特征 (50个点)
y = 2 * X + 3 + np.random.randn(50) * 2  # 目标值 (加噪声)

# 2. 参数初始化
w = np.random.randn()  # 权重
b = np.random.randn()  # 偏置
lr = 0.01              # 学习率
epochs = 1000          # 迭代次数

# 3. 训练 (梯度下降)
for i in range(epochs):
    # 前向传播 (预测值)
    y_pred = w * X + b
    
    # 计算损失 (均方误差 MSE)
    loss = np.mean((y_pred - y) ** 2)
    
    # 反向传播 (计算梯度)
    dw = np.mean(2 * (y_pred - y) * X)  # 对w的偏导
    db = np.mean(2 * (y_pred - y))      # 对b的偏导
    
    # 参数更新
    w -= lr * dw
    b -= lr * db
    
    # 每100次输出一次
    if i % 100 == 0:
        print(f"Epoch {i}: loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

# 4. 可视化结果
plt.scatter(X, y, label="Data")                  # 原始数据点
plt.plot(X, w * X + b, color="red", label="Fit") # 拟合直线
plt.legend()
plt.show()
