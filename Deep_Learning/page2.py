"""
页面2 ，展示回归
"""

import traceback

import matplotlib.pyplot as plt
import numpy as np


def forward(x_data, y_data, a_fn, lr=0.3, Epochs=1000, w=0, b=0):
    try:
        # 定义激活函数及其导数，方便未来扩展
        activation_functions = {
            '无': {'fn': lambda x: x, 'deriv': lambda x: 1},
            'sigmoid': {'fn': lambda x: 1 / (1 + np.exp(-x)), 'deriv': lambda a: a * (1 - a)}
        }

        # 确保选择的激活函数存在
        if a_fn not in activation_functions:
            raise ValueError(f"不支持的激活函数: {a_fn}")

        # 获取当前激活函数及其导数
        act_fn = activation_functions[a_fn]['fn']
        act_deriv = activation_functions[a_fn]['deriv']

        # 公共损失函数
        def loss_fn(y_true, y_pre):
            return np.mean((y_true - y_pre) ** 2)

        # 模型函数 (线性部分 + 激活函数)
        def model(w, x, b):
            return act_fn(w * x + b)


        # 初始化图形
        fig = plt.figure(f"Linear regression with {a_fn}", figsize=(12, 6))

        # 准备网格数据用于3D曲面和等高线图
        w_values = np.linspace(-20, 80, 100)
        b_values = np.linspace(-20, 80, 100)
        W, B = np.meshgrid(w_values, b_values)

        # 计算损失值网格
        loss_values = np.zeros_like(W)
        for i, w_v in enumerate(w_values):
            for j, b_v in enumerate(b_values):
                loss_values[j, i] = loss_fn(y_data, model(w_v, x_data, b_v))

        # 创建子图
        ax1 = fig.add_subplot(2, 2, 1)  # 散点图和拟合线
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")  # 3D损失曲面
        ax3 = fig.add_subplot(2, 2, 3)  # 等高线图
        ax4 = fig.add_subplot(2, 2, 4)  # 损失函数图

        # 初始化3D曲面和等高线图
        ax2.plot_surface(W, B, loss_values, cmap="viridis", alpha=0.8)
        ax3.contour(W, B, loss_values, cmap="viridis")

        # 存储训练过程数据
        epoch_lists = []
        loss_lists = []
        gd_path = []

        # 训练循环
        for epoch in range(Epochs):
            # 前向传播
            z = w * x_data + b
            y_pre = act_fn(z)
            loss = loss_fn(y_data, y_pre)
            # 反向传播计算梯度（链式法则）
            dedy = -2 * (y_data - y_pre)  # 损失对输出的导数
            dydz = act_deriv(y_pre)  # 激活函数对z的导数
            dzdw = x_data  # z对w的导数
            dzdb = 1  # z对b的导数
            # 计算梯度
            dw = np.mean(dedy * dydz * dzdw)
            db = np.mean(dedy * dydz * dzdb)
            # 更新参数
            w -= lr * dw
            b -= lr * db
            # 记录训练路径和损失
            gd_path.append((w, b))
            epoch_lists.append(epoch)
            loss_lists.append(loss)
            # 定期更新并返回图像
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"[{epoch + 1}/{Epochs}] w:{round(w, 3)} b:{round(b, 3)} Loss:{round(loss, 3)}")
                # 更新散点图和拟合线
                ax1.clear()
                ax1.scatter(x_data, y_data, color="b")
                x_min,x_max = x_data.min(),x_data.max()
                y_min,y_max = model(w,x_min,b), model(w,x_max,b)
                ax1.plot([x_min,x_max], [y_min,y_max], color="r")
                ax1.set_title(f"w:{round(w, 3)} b:{round(b, 3)} Loss:{round(loss, 3)}")

                # 更新3D曲面和梯度路径
                ax2.scatter(w, b, loss, color="black", s=20)
                if len(gd_path) > 0:
                    gd_w, gd_b = zip(*gd_path)
                    loss_path = [loss_fn(y_data, model(wv, x_data, bv)) for wv, bv in zip(gd_w, gd_b)]
                    ax2.plot(gd_w, gd_b, loss_path, color="black")


                # 更新等高线图
                ax3.clear()
                ax3.contour(W, B, loss_values, cmap="viridis")
                ax3.scatter(w, b, color="black", s=20)
                if len(gd_path) > 0:
                    gd_w, gd_b = zip(*gd_path)
                    ax3.plot(gd_w, gd_b)

                # 更新损失函数图
                ax4.clear()
                ax4.plot(epoch_lists, loss_lists)
                ax4.set_title("Loss over epochs")
                yield fig
    except (ValueError, ZeroDivisionError) as e:
        # 获取异常的详细信息
        print(f"捕获异常: {e}")
        # 打印异常发生的位置（文件名、行号等）
        print("异常发生在:")
        traceback.print_exc()  # 打印完整的异常追踪信息
def back_forward():
    pass
