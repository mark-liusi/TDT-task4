# export_torchscript.py
# 作用：将训练好的 PyTorch 模型导出为 TorchScript (.pt)，便于 C++ 加载
import torch
import torch.nn as nn
import torch.nn.functional as F
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

device= 'cpu'  # 建议导出 CPU 版，部署更通用；需要 CUDA 再在 C++ 侧 to(cuda)
model= ImprovedNet()
state= torch.load('mnist_fc_improved.pth', map_location=device)  # 若保存的是 state_dict
model.load_state_dict(state)
model.eval()

# dummy 输入要与训练的输入尺寸/通道一致，例如 1×1×28×28 或 1×3×H×W
example= torch.randn(1, 1, 28, 28)
traced= torch.jit.trace(model, example)
traced.save('model_ts.pt')
print('ok: model_ts.pt')
