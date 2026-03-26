import numpy as np
import matplotlib.pyplot as plt


def error(x, y, slope, intercept):
    return np.mean((slope * x + intercept - y) ** 2)


def plot_regression_line(x, y, slope, intercept):
    plt.scatter(x, y, color='blue', label='Original data')
    plt.plot(x, slope * x + intercept, color='red', label='Regression line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def linear_regression(x, y):
    """
    Linear Regression - 使用梯度下降求解
    """
    x_flat = x.flatten()
    y_flat = y.flatten()
    n = len(x_flat)

    # 超参数
    learning_rate = 0.01
    epochs = 1000

    # 初始化参数
    w = 0.0
    b = 0.0

    for epoch in range(epochs):
        y_pred = w * x_flat + b
        dw = (2 / n) * np.sum(x_flat * (y_pred - y_flat))
        db = (2 / n) * np.sum(y_pred - y_flat)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if epoch % 100 == 0:
            loss = np.mean((y_pred - y_flat) ** 2)
            print(f"Epoch {epoch}, Loss: {loss:.6f}, w: {w:.4f}, b: {b:.4f}")

    return w, b


def main():
    data = np.load('data.npy', allow_pickle=True).item()
    x = data['X']
    y = data['y']
    slope, intercept = linear_regression(x, y)
    plot_regression_line(x, y, slope, intercept)
    mse = error(x, y, slope, intercept)
    print(f"Final MSE: {mse:.6f}")


if __name__ == "__main__":
    main()