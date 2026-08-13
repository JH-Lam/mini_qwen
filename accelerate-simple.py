import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator

# 初始化
accelerator = Accelerator()

# 构建简单 MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


# 构造训练集和验证集
x_train = torch.randn(800, 100)
y_train = torch.randint(0, 10, (800,))
train_dataset = TensorDataset(x_train, y_train)

x_val = torch.randn(200, 100)
y_val = torch.randint(0, 10, (200,))
val_dataset = TensorDataset(x_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 初始化模型、优化器、损失函数
model = MLP()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 交给 accelerator 管理
model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

best_val_acc = 0.0  # 记录最佳验证集准确率

# 训练
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, targets = batch
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

    val_acc = correct / total

    # 保存最优模型
    if accelerator.is_main_process:
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            accelerator.save(accelerator.get_state_dict(model), "best_model.pt")
            print(f"Best model saved at Epoch {epoch + 1} with Val Acc: {val_acc:.4f}")

    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
