import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# ==================== 1. 生成合成数据 ====================
print("Generating synthetic data...")
true_w = 2.0
true_b = 3.0
X = np.random.randn(100, 1)
noise = np.random.randn(100, 1) * 0.5
y = true_w * X + true_b + noise

train_size = 80
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Train size: {train_size}, Test size: {100 - train_size}")

# ==================== 2. NumPy 手写线性回归 ====================
print("\nTraining NumPy linear regression...")

w = np.random.randn(1, 1)
b = np.zeros((1, 1))

learning_rate = 0.1
num_epochs = 100
losses_np = []

for epoch in range(num_epochs):
    y_pred = X_train @ w + b
    loss = np.mean((y_pred - y_train) ** 2)
    losses_np.append(loss)

    grad_w = (2 / train_size) * X_train.T @ (y_pred - y_train)
    grad_b = (2 / train_size) * np.sum(y_pred - y_train)

    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.6f}")

print(f"\nNumPy results: w = {w[0][0]:.4f}, b = {b[0][0]:.4f}")

# ==================== 3. PyTorch 线性回归 ====================
print("\nTraining PyTorch linear regression...")

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

losses_pt = []
for epoch in range(num_epochs):
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    losses_pt.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")

pt_w = model.linear.weight.item()
pt_b = model.linear.bias.item()
print(f"\nPyTorch results: w = {pt_w:.4f}, b = {pt_b:.4f}")

# ==================== 4. 打印对比结果 ====================
print("\n" + "=" * 50)
print("Parameter Comparison:")
print("=" * 50)
print(f"{'Method':<12} {'w':<10} {'b':<10} {'w_error':<10} {'b_error':<10}")
print("-" * 50)
print(f"{'NumPy':<12} {w[0][0]:.4f}     {b[0][0]:.4f}     {abs(w[0][0] - true_w):.4f}      {abs(b[0][0] - true_b):.4f}")
print(f"{'PyTorch':<12} {pt_w:.4f}     {pt_b:.4f}     {abs(pt_w - true_w):.4f}      {abs(pt_b - true_b):.4f}")
print(f"{'True':<12} {true_w:.4f}     {true_b:.4f}     {'-':<10} {'-':<10}")

# ==================== 5. 测试集评估 ====================
y_pred_np = X_test @ w + b
mse_np = np.mean((y_pred_np - y_test) ** 2)

with torch.no_grad():
    y_pred_pt = model(X_test_tensor).numpy()
    mse_pt = np.mean((y_pred_pt - y_test) ** 2)

print(f"\nTest MSE:")
print(f"NumPy:  {mse_np:.6f}")
print(f"PyTorch: {mse_pt:.6f}")

# ==================== 6. 可视化（使用英文标签）====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss Curves
ax1 = axes[0, 0]
ax1.plot(losses_np, label='NumPy', linewidth=2, color='blue')
ax1.plot(losses_pt, label='PyTorch', linewidth=2, linestyle='--', color='red')
ax1.set_title('Loss Curves (MSE)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')  # 使用对数坐标，更清晰显示下降过程

# Plot 2: Data and Fitted Lines
ax2 = axes[0, 1]
ax2.scatter(X_train, y_train, alpha=0.5, label='Train Data', s=20, color='gray')
ax2.scatter(X_test, y_test, alpha=0.7, label='Test Data', s=30, marker='^', color='orange')
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line_np = x_line * w[0][0] + b[0][0]
y_line_pt = x_line * pt_w + pt_b
y_line_true = true_w * x_line + true_b
ax2.plot(x_line, y_line_np, 'b-', label='NumPy Fit', linewidth=2)
ax2.plot(x_line, y_line_pt, 'r--', label='PyTorch Fit', linewidth=2)
ax2.plot(x_line, y_line_true, 'k:', label='True Line', linewidth=2)
ax2.set_title('Data Distribution and Fitted Lines', fontsize=14, fontweight='bold')
ax2.set_xlabel('X', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: Predictions vs True Values
ax3 = axes[1, 0]
ax3.scatter(y_test, y_pred_np, alpha=0.6, label='NumPy', s=30, color='blue')
ax3.scatter(y_test, y_pred_pt, alpha=0.6, label='PyTorch', s=30, marker='^', color='red')
min_val = min(y_test.min(), y_pred_np.min(), y_pred_pt.min())
max_val = max(y_test.max(), y_pred_np.max(), y_pred_pt.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal (y=x)', linewidth=2)
ax3.set_title('Predictions vs True Values (Test Set)', fontsize=14, fontweight='bold')
ax3.set_xlabel('True Values', fontsize=12)
ax3.set_ylabel('Predictions', fontsize=12)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Plot 4: Residual Distribution
ax4 = axes[1, 1]
residuals_np = y_test - y_pred_np
residuals_pt = y_test - y_pred_pt
ax4.hist(residuals_np, bins=20, alpha=0.5, label='NumPy Residuals', density=True, color='blue', edgecolor='black')
ax4.hist(residuals_pt, bins=20, alpha=0.5, label='PyTorch Residuals', density=True, color='red', edgecolor='black')
ax4.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Zero Error')
ax4.set_title('Residual Distribution', fontsize=14, fontweight='bold')
ax4.set_xlabel('Residual', fontsize=12)
ax4.set_ylabel('Density', fontsize=12)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVisualization complete! Figure saved as 'linear_regression_results.png'")

# ==================== 7. 额外：打印训练统计 ====================
print("\n" + "=" * 50)
print("Training Statistics:")
print("=" * 50)
print(f"Final Loss (NumPy):  {losses_np[-1]:.6f}")
print(f"Final Loss (PyTorch): {losses_pt[-1]:.6f}")
print(f"Parameter Difference: w_diff = {abs(w[0][0] - pt_w):.6f}, b_diff = {abs(b[0][0] - pt_b):.6f}")
print(f"Test MSE Difference: {abs(mse_np - mse_pt):.8f}")