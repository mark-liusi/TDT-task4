import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Use default English fonts
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 加载MNIST数据集
print("正在加载MNIST数据集...")
try:
    # 方法1: 使用keras加载(推荐)
    from tensorflow import keras

    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = (
        keras.datasets.mnist.load_data()
    )
    # 将28x28的图像展平为784维向量
    X = np.vstack([X_train_raw.reshape(-1, 784), X_test_raw.reshape(-1, 784)])
    y = np.hstack([y_train_raw, y_test_raw])
    print(f"成功加载MNIST数据集: {X.shape[0]}个样本")
except ImportError:
    # 方法2: 使用torchvision加载
    try:
        from torchvision import datasets
        import torch

        train_dataset = datasets.MNIST(root="./data", train=True, download=True)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True)
        X_train_raw = train_dataset.data.numpy().reshape(-1, 784)
        y_train_raw = train_dataset.targets.numpy()
        X_test_raw = test_dataset.data.numpy().reshape(-1, 784)
        y_test_raw = test_dataset.targets.numpy()
        X = np.vstack([X_train_raw, X_test_raw])
        y = np.hstack([y_train_raw, y_test_raw])
        print(f"成功加载MNIST数据集: {X.shape[0]}个样本")
    except ImportError:
        print("错误: 需要安装 tensorflow 或 pytorch")
        print("请运行: pip install tensorflow 或 pip install torch torchvision")
        exit(1)

# Data standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================== 任务(1): 二分类模型 ====================
print("\n" + "=" * 50)
print("任务(1): 二分类模型 (>=5 vs <5)")
print("=" * 50)

# Filter digits >= 5 or < 5
binary_mask = (y >= 5) | (y < 5)  # All data
X_binary = X_scaled[binary_mask]
y_binary = (y[binary_mask] >= 5).astype(int)  # 1 means >=5, 0 means <5

# Split train and test set
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42
)

# Train logistic regression model
lr_binary = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
lr_binary.fit(X_train_bin, y_train_bin)

# Evaluate model
y_pred_bin = lr_binary.predict(X_test_bin)
acc_bin = accuracy_score(y_test_bin, y_pred_bin)
print(f"\n二分类模型准确率: {acc_bin:.4f}")
print("\n分类报告:")
print(classification_report(y_test_bin, y_pred_bin, target_names=["<5", ">=5"]))


# ==================== 任务(2): 多分类模型 ====================
print("\n" + "=" * 50)
print("任务(2): 多分类模型 (分段分类)")
print("=" * 50)


# Classify digits into 4 groups: [1,3], [4,6], [7,9], 0 alone
def classify_digit(digit):
    if 1 <= digit <= 3:
        return 0
    elif 4 <= digit <= 6:
        return 1
    elif 7 <= digit <= 9:
        return 2
    else:  # 0
        return 3


y_multi = np.array([classify_digit(d) for d in y])

# Split train and test set
X_train_mul, X_test_mul, y_train_mul, y_test_mul = train_test_split(
    X_scaled, y_multi, test_size=0.2, random_state=42
)

# Train multi-class logistic regression model
lr_multi = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
lr_multi.fit(X_train_mul, y_train_mul)

# Evaluate model
y_pred_mul = lr_multi.predict(X_test_mul)
acc_mul = accuracy_score(y_test_mul, y_pred_mul)
print(f"\n多分类模型准确率: {acc_mul:.4f}")
print("\n分类报告:")
print(
    classification_report(
        y_test_mul, y_pred_mul, target_names=["[1-3]", "[4-6]", "[7-9]", "[0]"]
    )
)


# ==================== 任务(3): 特征重要度分析 ====================
print("\n" + "=" * 50)
print("任务(3): 特征重要度分析")
print("=" * 50)

# Get weight coefficients for each class
coefficients = lr_multi.coef_  # shape: (n_classes, n_features)

# Calculate feature importance (using mean of absolute values)
feature_importance = np.mean(np.abs(coefficients), axis=0)

# Find top important features
top_k = 100
top_features_idx = np.argsort(feature_importance)[-top_k:]
print(f"\n最重要的{top_k}个特征已选出")
print(
    f"特征重要度范围: {feature_importance.min():.6f} ~ {feature_importance.max():.6f}"
)

# Visualize top features with bar chart
plt.figure(figsize=(14, 6))
top_50_idx = np.argsort(feature_importance)[-50:]
plt.bar(
    range(50),
    feature_importance[top_50_idx],
    color="steelblue",
    edgecolor="black",
    alpha=0.7,
)
plt.xlabel("Feature Rank", fontsize=12)
plt.ylabel("Feature Importance", fontsize=12)
plt.title("Top 50 Most Important Features (Bar Chart)", fontsize=14, fontweight="bold")
plt.xticks(range(0, 50, 5))
plt.grid(axis="y", alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(
    "/home/liusi/文档/code/TDT-task4/feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("特征重要度柱状图已保存!")


# ==================== 任务(4): 使用特征选择重建模型 ====================
print("\n" + "=" * 50)
print("任务(4): 使用特征选择重建模型")
print("=" * 50)

# Try different numbers of features
feature_counts = [50, 100, 200, 300, 400]
accuracies = []

for k in feature_counts:
    # Select top-k most important features
    selected_features = np.argsort(feature_importance)[-k:]

    X_train_selected = X_train_mul[:, selected_features]
    X_test_selected = X_test_mul[:, selected_features]

    # Train new model
    lr_selected = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    lr_selected.fit(X_train_selected, y_train_mul)

    # Evaluate
    y_pred_selected = lr_selected.predict(X_test_selected)
    acc_selected = accuracy_score(y_test_mul, y_pred_selected)
    accuracies.append(acc_selected)

    print(f"使用前{k}个特征的准确率: {acc_selected:.4f}")

# Visualize relationship between feature count and accuracy
plt.figure(figsize=(10, 6))
plt.plot(
    feature_counts, accuracies, marker="o", linewidth=2, markersize=8, color="steelblue"
)
plt.axhline(
    y=acc_mul,
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Full features accuracy: {acc_mul:.4f}",
)
plt.xlabel("Number of Features", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.title(
    "Impact of Feature Selection on Model Performance", fontsize=14, fontweight="bold"
)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig(
    "/home/liusi/文档/code/TDT-task4/feature_selection_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("特征选择对比图已保存!")


# ==================== 任务(5): 模型对比与分析 ====================
print("\n" + "=" * 50)
print("任务(5): 模型对比与分析")
print("=" * 50)

print("\n模型性能对比:")
print(f"完整特征模型准确率: {acc_mul:.4f}")
print(
    f"最佳特征选择模型准确率: {max(accuracies):.4f} (使用{feature_counts[np.argmax(accuracies)]}个特征)"
)

print("\n分析与思考:")
print("1. 特征选择的优势:")
print("   - 降低模型复杂度,减少过拟合风险")
print("   - 加快训练和预测速度")
print("   - 提高模型可解释性")

print("\n2. 观察到的现象:")
print(
    f"   - 使用约{feature_counts[np.argmax(accuracies)]}个特征可达到接近完整模型的性能"
)
print(
    f"   - 这意味着约{(1-feature_counts[np.argmax(accuracies)]/784)*100:.1f}%的特征是冗余的"
)

print("\n3. 特征重要度分布:")
print("   - 中心区域的像素通常更重要(包含数字主体)")
print("   - 边缘区域的像素重要度较低(通常为背景)")

# Save model comparison results
comparison_results = {
    "模型": [
        "二分类",
        "多分类(完整特征)",
        f"多分类({feature_counts[np.argmax(accuracies)]}特征)",
    ],
    "准确率": [acc_bin, acc_mul, max(accuracies)],
    "特征数": [784, 784, feature_counts[np.argmax(accuracies)]],
}

print("\n" + "=" * 50)
print("所有任务完成!")
print("=" * 50)
