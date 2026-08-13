import torch
import torch.nn as nn
import matplotlib.pyplot as plt

'''
 在 PyTorch 中使用 SGD（随机梯度下降）优化器非常直接，它是最基础也最常用的优化器之一。下面我会用通俗易懂的方式讲解，并提供完整的代码示例。

SGD 优化器的核心概念
SGD 的工作原理很简单：

计算模型参数的梯度（损失函数对参数的偏导数）
按照梯度的反方向更新参数，更新幅度由学习率控制
可以添加动量（momentum）来加速收敛，减少震荡
使用步骤
定义模型
定义损失函数
初始化 SGD 优化器，指定要优化的参数和超参数
在训练循环中：
前向传播计算预测值
计算损失
反向传播计算梯度
用优化器更新参数

————————————————
版权声明：本文为CSDN博主「Aurora-Orion」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/2301_79556402/article/details/151121727

'''
# 1. 准备数据
# 生成一些带噪声的线性数据 y = 3x + 2 + 噪声
x = torch.randn(100, 1) * 10  # 100个随机数
y = 3 * x + 2 + torch.randn(100, 1) * 3  # 加入噪声


# 2. 定义模型：简单的线性模型 y = wx + b
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个线性层，输入1维，输出1维
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)


# 初始化模型
model = LinearModel()

# 3. 定义损失函数：均方误差
loss_function = nn.MSELoss()

# 4. 定义SGD优化器
# 参数说明：
# - model.parameters()：需要优化的模型参数
# - lr=0.01：学习率，控制更新步长
# - momentum=0.9：动量，加速收敛，减少震荡（可选）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 5. 训练模型
epochs = 100  # 训练轮次
losses = []  # 记录损失变化

for epoch in range(epochs):
    # 前向传播：计算模型预测值
    y_pred = model(x)

    # 计算损失
    loss = loss_function(y_pred, y)
    losses.append(loss.item())

    # 清空之前的梯度（非常重要！）
    optimizer.zero_grad()

    # 反向传播：计算梯度
    loss.backward()

    # 用SGD优化器更新参数
    optimizer.step()

    # 每10轮打印一次信息
    if (epoch + 1) % 10 == 0:
        # 获取当前的权重和偏置
        w, b = model.linear.weight.item(), model.linear.bias.item()
        print(f"轮次: {epoch + 1}, 损失: {loss.item():.4f}, 权重: {w:.4f}, 偏置: {b:.4f}")

# 6. 结果可视化
plt.figure(figsize=(12, 5))

# 绘制数据点和拟合直线
plt.subplot(1, 2, 1)
plt.scatter(x.numpy(), y.numpy(), label='数据点')
plt.plot(x.numpy(), model(x).detach().numpy(), 'r-', label=f'拟合线: y={w:.2f}x+{b:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# 绘制损失变化曲线
plt.subplot(1, 2, 2)
plt.plot(range(epochs), losses)
plt.xlabel('轮次')
plt.ylabel('损失值')
plt.title('损失变化曲线')

plt.tight_layout()
plt.show()
