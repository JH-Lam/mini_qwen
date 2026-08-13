import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 1. 准备更贴近真实场景的数据
np.random.seed(42)
torch.manual_seed(42)

# 生成非线性可分数据（模拟更复杂场景）
x = torch.linspace(-10, 10, 200).view(-1, 1)  # 200个从-10到10的点
y = 0.5 * x ** 2 + 2 * x + 3 + torch.randn_like(x) * 5  # 二次函数+噪声


# 2. 定义一个稍复杂的模型（包含非线性层）
class NonLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 10)  # 第一层：1→10维
        self.activation = nn.ReLU()  # 非线性激活
        self.layer2 = nn.Linear(10, 1)  # 第二层：10→1维

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        return self.layer2(x)


# 初始化两个模型，用于对比不同动量的效果
model_with_momentum = NonLinearModel()
model_no_momentum = NonLinearModel()  # 复制结构但参数独立

# 3. 定义损失函数（均方误差）
loss_fn = nn.MSELoss()

# 4. 初始化两个SGD优化器（带动量 vs 无动量）
# 带动量的SGD（适合复杂模型）
optimizer_with_momentum = torch.optim.SGD(
    model_with_momentum.parameters(),
    lr=0.001,  # 学习率
    momentum=0.9,  # 动量（0.9是常用值）
    weight_decay=1e-4  # 权重衰减（L2正则化，可选）
)

# 无动量的SGD（基础版）
optimizer_no_momentum = torch.optim.SGD(
    model_no_momentum.parameters(),
    lr=0.001,
    momentum=0.0  # 关闭动量
)

# 5. 定义学习率调度器（动态调整学习率，非常实用！）
# 每50轮将学习率乘以0.5
scheduler_with_momentum = torch.optim.lr_scheduler.StepLR(
    optimizer_with_momentum,
    step_size=50,
    gamma=0.5
)
scheduler_no_momentum = torch.optim.lr_scheduler.StepLR(
    optimizer_no_momentum,
    step_size=50,
    gamma=0.5
)

# 6. 训练两个模型并记录损失
epochs = 300
losses_with_momentum = []
losses_no_momentum = []

for epoch in range(epochs):
    # 训练带动量的模型
    model_with_momentum.train()  # 切换到训练模式
    y_pred_m = model_with_momentum(x)
    loss_m = loss_fn(y_pred_m, y)
    losses_with_momentum.append(loss_m.item())

    optimizer_with_momentum.zero_grad()  # 清空梯度
    loss_m.backward()  # 反向传播
    optimizer_with_momentum.step()  # 更新参数
    scheduler_with_momentum.step()  # 调整学习率

    # 训练无动量的模型（相同步骤）
    model_no_momentum.train()
    y_pred_nm = model_no_momentum(x)
    loss_nm = loss_fn(y_pred_nm, y)
    losses_no_momentum.append(loss_nm.item())

    optimizer_no_momentum.zero_grad()
    loss_nm.backward()
    optimizer_no_momentum.step()
    scheduler_no_momentum.step()

    # 每50轮打印信息
    if (epoch + 1) % 50 == 0:
        print(f"轮次 {epoch + 1}/{epochs}")
        print(f"带动量损失: {loss_m.item():.4f} | 无动量损失: {loss_nm.item():.4f}")
        print(f"当前学习率: {optimizer_with_momentum.param_groups[0]['lr']:.6f}\n")

# 7. 结果可视化
plt.figure(figsize=(14, 6))

# 绘制拟合效果对比
plt.subplot(1, 2, 1)
plt.scatter(x.numpy(), y.numpy(), alpha=0.5, label='ori data')
plt.plot(x.numpy(), model_with_momentum(x).detach().numpy(), 'r-', linewidth=2, label='sgd w/ momentum')
plt.plot(x.numpy(), model_no_momentum(x).detach().numpy(), 'b--', linewidth=2, label='sgd w/o momentum')
plt.title('Fitness comparision')
plt.legend()

# 绘制损失曲线对比
plt.subplot(1, 2, 2)
plt.plot(range(epochs), losses_with_momentum, 'r-', label='with momentum (momentum=0.9)')
plt.plot(range(epochs), losses_no_momentum, 'b--', label='w/o momentum (momentum=0.0)')
plt.axvline(x=50, color='gray', linestyle=':', label='lr decray points')
plt.axvline(x=100, color='gray', linestyle=':')
plt.axvline(x=150, color='gray', linestyle=':')
plt.title('loss cure')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.tight_layout()
plt.show()
