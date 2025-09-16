
import queue
import threading
import traceback

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles, make_moons, make_regression, make_friedman1
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA


from sklearn.preprocessing import StandardScaler

class DrawNet:
    """
    网络可视化
    """

    def __init__(self, architecture, node_radius=0.03, weight_threshold=0.1):
        self.architecture = architecture
        self.node_radius = node_radius
        self.node_pos = {}
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.axis("off")
        # 权重阈值（绝对值小于这个值的连接显示灰色，可按需调整）
        self.weight_threshold = weight_threshold

        # 创建一个created_figures用于最后清除plt内存
        self.created_figures = []  # 用于跟踪创建的图形,用于后面的内存资源清理
        self.created_figures.append(self.fig)  # 添加到created_figures
        plt.ion()
        plt.show(block=False)

    def compute_node_positions(self):
        """
        l表示第几层，i表示层内的第几个神经元
        :return:
        """
        n_layers = len(self.architecture)
        x_spacing = 1.0 / (n_layers - 1)
        for l, n_nodes in enumerate(self.architecture):
            y_spacing = 1.0 / (n_nodes + 1)
            for i in range(n_nodes):
                x = l * x_spacing
                y = 1 - (i + 1) * y_spacing
                self.node_pos[(l, i)] = (x, y)

    def weight_to_color(self, weight):
        """用于根据权重切换颜色"""
        # 1. 先判断权重是否过小（绝对值 < 阈值）
        if abs(weight) < self.weight_threshold:
            # 小权重：灰色（透明度固定0.5，避免完全看不见）
            return (0.5, 0.5, 0.5, 0.5)  # (R,G,B,alpha)，纯灰色
        # 2. 权重足够大：保持原有红蓝渐变逻辑
        alpha = min(1, abs(weight))  # 颜色深浅随权重绝对值变化
        if weight >= 0:
            return (1, 0, 0, alpha)  # 正权重→红色
        else:
            return (0, 0, 1, alpha)  # 负权重→蓝色

    def draw(self, W_list):
        """W_list = [W1, W2, W3]"""
        self.ax.clear()
        self.ax.axis("off")
        self.compute_node_positions()

        # 画连线 + 标注权重
        for l in range(len(self.architecture) - 1):
            W = W_list[l]
            n_from, n_to = W.shape
            for i in range(n_from):
                for j in range(n_to):
                    x1, y1 = self.node_pos[(l, i)]
                    x2, y2 = self.node_pos[(l + 1, j)]
                    # 颜色会自动调用weight_to_color，小权重显示灰色
                    color = self.weight_to_color(W[i, j])
                    # 小权重连线粗细固定（避免太细看不见），大权重按原有逻辑变粗
                    if abs(W[i, j]) < self.weight_threshold:
                        lw = 0.8  # 小权重连线：固定中等粗细，确保可见
                    else:
                        lw = 0.5 + 2 * abs(W[i, j])  # 大权重连线：随绝对值变粗
                    self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, zorder=1)

        # 画节点
        for (l, i), (x, y) in self.node_pos.items():
            if l < len(self.architecture) - 1:
                avg_w = np.mean(np.abs(W_list[l][i, :]))
                node_color = (1, 0, 0, min(1, avg_w))
            else:
                node_color = (0.8, 0.8, 0.8, 1)
            self.ax.add_patch(plt.Circle((x, y), self.node_radius, color=node_color, zorder=3))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return self.fig

    # 添加图形清理方法
    def close_figures(self):
        """关闭所有创建的图形，释放内存"""
        for fig in self.created_figures:
            plt.close(fig)
        self.created_figures = []  # 用于跟踪创建的图形

    # 退出时自动清理
    def __del__(self):
        self.close_figures()


class DrawEffect:
    """
    绘制神经网络旁边的效果图
    """

    def __init__(self, X, y, model, dataset_type):
        self.pca = PCA(n_components=2)  # 先降维，二维显示
        self.X = self.pca.fit_transform(X)
        self.y = y
        self.model = model
        self.dataset_type = dataset_type
        self.created_figures = []  # 用于跟踪创建的图形,用于后面的内存资源清理

        # 三维拟合图
        self.fit_surface = plt.figure(figsize=(8, 6))
        self.fit_surface_ax = self.fit_surface.add_subplot(2, 1, 1, projection='3d')
        self.fit_surface_ax.scatter(self.X[:, 0], self.X[:, 1], self.y, )  # 先画个散点
        self.created_figures.append(self.fit_surface_ax)  # 添加到跟踪列表

        # 决策边界
        self.boundary_fig = plt.figure(figsize=(8, 6))
        self.boundary_ax = self.boundary_fig.add_subplot(2, 1, 1)
        self.created_figures.append(self.boundary_fig)  # 添加到跟踪列表
        # 损失函数
        if self.dataset_type == "回归":
            self.loss_curve_ax = self.fit_surface.add_subplot(2, 1, 2)
        else:
            self.loss_curve_ax = self.boundary_fig.add_subplot(2, 1, 2)

    def draw_decision_boundary(self):
        "绘制决策边界"
        X_min, X_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1  # 根据第一个主要成分定义X轴范围。+1和-1防止样本正好在边界上
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1  # 根据第二个主要成分定义Y轴范围。
        xx, yy = np.meshgrid(
            np.arange(X_min, X_max, 0.5),  # 根据范围创建等差数列作为一个个网格
            np.arange(y_min, y_max, 0.5),  # 根据范围创建等差数列作为一个个网格
        )
        with torch.no_grad():
            with torch.no_grad():
                grid_points = np.c_[xx.ravel(), yy.ravel()]
                grid_points = self.pca.inverse_transform(grid_points)
                Z = self.model(torch.tensor(grid_points).float())
                Z = torch.argmax(Z, dim=1).numpy()  # Z需要的其实是预测结果，用argmax返回所有类别里面概率最大的那个类，这样就得到了预测结果

        # Z 变形回原形状用于标记每个网格点的类。
        Z = Z.reshape(xx.shape)

        self.boundary_ax.clear()
        # 绘制
        # 绘制决策边界

        self.boundary_ax.contourf(xx, yy, Z, )  # xx和yy表示了每个网格点，Z表示了每个网格点的类型，故第三个参数需要和xx和yy形状相同
        self.boundary_ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, edgecolors="k")  # c=y表示用目标值进行着色，逐个着色，标签不同颜色不同

    def draw_fit_surface(self):
        """绘制3D拟合曲面"""
        # 获取x1和x2的取值范围
        x1_min, x1_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        x2_min, x2_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1

        # 创建网格数据
        x1_values = np.linspace(x1_min, x1_max, 50)
        x2_values = np.linspace(x2_min, x2_max, 50)
        x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)

        # 将网格数据展平并转换为适合模型输入的格式
        grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]
        # 使用模型进行预测
        with torch.no_grad():  # 关闭梯度计算，提高效率
            # 将NumPy数组转换为PyTorch张量
            input_tensor = torch.tensor(grid_points, dtype=torch.float32)
            # 应用PCA转换转为原维度
            input_tensor = torch.tensor(self.pca.inverse_transform(input_tensor.numpy()), dtype=torch.float32)
            # 模型预测
            y_pred = self.model(input_tensor)
            # 将预测结果转换回NumPy数组并重塑为网格形状
            y_grid = y_pred.numpy().reshape(x1_grid.shape)

        # 清除当前轴并设置3D投影
        self.fit_surface_ax.clear()

        # 绘制3D曲面
        surf = self.fit_surface_ax.plot_surface(
            x1_grid, x2_grid, y_grid,
            cmap='viridis', alpha=0.7
        )
        # 在画曲面之后，把真实散点叠上去
        self.fit_surface_ax.scatter(
            self.X[:, 0], self.X[:, 1], self.y,
            color='black', s=20, label='true samples')
        # 设置坐标轴标签
        self.fit_surface_ax.set_xlabel('X1')
        self.fit_surface_ax.set_ylabel('X2')
        self.fit_surface_ax.set_zlabel('Y')

        # 添加颜色条以表示高度
        self.fit_surface.colorbar(surf, ax=self.fit_surface_ax, shrink=0.5, aspect=5)

        return self.fit_surface

    def draw_loss_curve(self, epochs_list, loss_list):
        """
        绘制损失下降曲线
        :param epochs_list:
        :param loss_list:
        :return:
        """
        self.loss_curve_ax.plot(epochs_list, loss_list)

    # 图形清理
    def close_figures(self):
        """关闭所有创建的图形，释放内存"""
        for fig in self.created_figures:
            plt.close(fig)
        self.created_figures = []

    # 退出时自动清理
    def __del__(self):
        self.close_figures()

