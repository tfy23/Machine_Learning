import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，保证结果可复现
np.random.seed(42)

# ==================== 1. 生成合成数据 ====================
print("生成合成数据...")
# 真实参数: y = 2*x + 3 + 噪声
true_w = 2.0
true_b = 3.0
X = np.random.randn(100, 1)  # 100个样本，1个特征
noise = np.random.randn(100, 1) * 0.5  # 噪声
y = true_w * X + true_b + noise

# 划分训练集和测试集
train_size = 80
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"训练集大小: {train_size}, 测试集大小: {100 - train_size}")

# ==================== 2. NumPy 手写线性回归 ====================
print("\n开始 NumPy 线性回归训练...")

# 初始化参数
w = np.random.randn(1, 1)  # 权重
b = np.zeros((1, 1))  # 偏置

# 超参数
learning_rate = 0.1
num_epochs = 100

# 记录损失
losses_np = []

# 训练循环
for epoch in range(num_epochs):
    # 前向传播：计算预测值
    y_pred = X_train @ w + b

    # 计算损失 (均方误差 MSE)
    loss = np.mean((y_pred - y_train) ** 2)
    losses_np.append(loss)

    # 计算梯度
    # dL/dw = (2/N) * X^T @ (y_pred - y)
    # dL/db = (2/N) * sum(y_pred - y)
    grad_w = (2 / train_size) * X_train.T @ (y_pred - y_train)
    grad_b = (2 / train_size) * np.sum(y_pred - y_train)

    # 更新参数
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    # 每10个epoch打印一次
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}")

print(f"\nNumPy 训练完成！")
print(f"学习到的参数: w = {w[0][0]:.4f}, b = {b[0][0]:.4f}")
print(f"真实参数: w = {true_w}, b = {true_b}")

# ==================== 3. PyTorch 线性回归（对比）====================
import torch
import torch.nn as nn
import torch.optim as optim

print("\n开始 PyTorch 线性回归训练...")

# 将数据转为 PyTorch 张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)


# 定义线性模型
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression()

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练
losses_pt = []
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    losses_pt.append(loss.item())

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 获取 PyTorch 学习到的参数
pt_w = model.linear.weight.item()
pt_b = model.linear.bias.item()
print(f"\nPyTorch 训练完成！")
print(f"学习到的参数: w = {pt_w:.4f}, b = {pt_b:.4f}")
print(f"真实参数: w = {true_w}, b = {true_b}")

# ==================== 4. 对比结果 ====================
print("\n" + "=" * 50)
print("参数对比:")
print("=" * 50)
print(f"{'方法':<15} {'w':<12} {'b':<12} {'w误差':<12} {'b误差':<12}")
print("-" * 50)
print(
    f"{'NumPy':<15} {w[0][0]:.4f}      {b[0][0]:.4f}      {abs(w[0][0] - true_w):.4f}       {abs(b[0][0] - true_b):.4f}")
print(f"{'PyTorch':<15} {pt_w:.4f}      {pt_b:.4f}      {abs(pt_w - true_w):.4f}       {abs(pt_b - true_b):.4f}")
print(f"{'真实值':<15} {true_w:.4f}      {true_b:.4f}      {'-':<12} {'-':<12}")

# ==================== 5. 在测试集上评估 ====================
# NumPy 预测
y_pred_np = X_test @ w + b
mse_np = np.mean((y_pred_np - y_test) ** 2)

# PyTorch 预测
with torch.no_grad():
    y_pred_pt = model(X_test_tensor).numpy()
    mse_pt = np.mean((y_pred_pt - y_test) ** 2)

print(f"\n测试集 MSE:")
print(f"NumPy:  {mse_np:.4f}")
print(f"PyTorch: {mse_pt:.4f}")

# ==================== 6. 可视化 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1：Loss 下降曲线对比
ax1 = axes[0, 0]
ax1.plot(losses_np, label='NumPy', linewidth=2)
ax1.plot(losses_pt, label='PyTorch', linewidth=2, linestyle='--')
ax1.set_title('Loss下降曲线对比', fontsize=12)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2：数据分布与拟合直线
ax2 = axes[0, 1]
ax2.scatter(X_train, y_train, alpha=0.5, label='训练数据', s=20)
ax2.scatter(X_test, y_test, alpha=0.7, label='测试数据', s=30, marker='^')
# 画拟合直线
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line_np = x_line * w[0][0] + b[0][0]
y_line_pt = x_line * pt_w + pt_b
ax2.plot(x_line, y_line_np, 'r-', label='NumPy拟合', linewidth=2)
ax2.plot(x_line, y_line_pt, 'g--', label='PyTorch拟合', linewidth=2)
ax2.plot(x_line, true_w * x_line + true_b, 'b:', label='真实直线', linewidth=2)
ax2.set_title('数据分布与拟合结果', fontsize=12)
ax2.set_xlabel('X')
ax2.set_ylabel('y')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3：预测值 vs 真实值（测试集）
ax3 = axes[1, 0]
ax3.scatter(y_test, y_pred_np, alpha=0.7, label='NumPy', s=30)
ax3.scatter(y_test, y_pred_pt, alpha=0.7, label='PyTorch', s=30, marker='^')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', label='理想线 (y=x)', linewidth=2)
ax3.set_title('预测值 vs 真实值（测试集）', fontsize=12)
ax3.set_xlabel('真实值')
ax3.set_ylabel('预测值')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4：残差分布
ax4 = axes[1, 1]
residuals_np = y_test - y_pred_np
residuals_pt = y_test - y_pred_pt
ax4.hist(residuals_np, bins=20, alpha=0.5, label='NumPy 残差', density=True)
ax4.hist(residuals_pt, bins=20, alpha=0.5, label='PyTorch 残差', density=True)
ax4.axvline(x=0, color='r', linestyle='--', linewidth=1)
ax4.set_title('残差分布', fontsize=12)
ax4.set_xlabel('残差')
ax4.set_ylabel('密度')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n可视化完成！")