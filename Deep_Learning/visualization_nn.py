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


def generate_data(dataset_type, n_samples=200, base_n_features=5, selected_derived_features=None):
    """
    生成数据,并可以选择增加衍生特征，衍生特征导致的维度问题在UI界面中用户输入时自动维护
    :param dataset_type:
    :param n_samples:
    :return:
    """

    if dataset_type == "分类":
        X, y = make_classification(n_samples=n_samples, n_features=base_n_features,
                                   n_redundant=0, n_informative=2,
                                   n_clusters_per_class=1, random_state=42)
    elif dataset_type == "圆形":
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
    elif dataset_type == "月形":
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    elif dataset_type == "回归":
        X, y = make_regression(n_samples=1000, n_features=base_n_features, n_informative=2,
                               noise=5.0, bias=1, random_state=42)
    elif dataset_type == "回归2.0":
        X, y = make_friedman1(n_samples=1000, n_features=base_n_features,
                              noise=5.0, random_state=42)
    else:
        X, y = make_moons(n_samples=base_n_features)

    # 记录一下原始的X和y，等会算衍生特征的时候就用这个，不然会混淆
    base_X, base_y = X, y

    if "交叉项" in selected_derived_features:
        # 获取当前特征数量
        n = base_X.shape[1]
        # 只计算i < j的组合，避免重复
        for i in range(n):
            for j in range(i + 1, n):
                # 计算交叉项（第i列与第j列的乘积）
                cross_term = base_X[:, i] * base_X[:, j]
                # 转换为列向量并拼接
                X = np.hstack([X, cross_term.reshape(-1, 1)])
    if "平方项" in selected_derived_features:
        square_terms = base_X ** 2
        # 将原始特征与平方项横向拼接
        X = np.hstack([X, square_terms])

    if "正弦项" in selected_derived_features:
        sin_terms = np.sin(base_X)
        # 将原始特征与平方项横向拼接
        X = np.hstack([X, sin_terms])

    if "余弦项" in selected_derived_features:
        cos_terms = np.cos(base_X)
        # 将原始特征与平方项横向拼接
        X = np.hstack([X, cos_terms])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # 归一化、不然可能会梯度爆炸
    return X, y


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


class Net(nn.Module):
    """
    定义网络
    """

    def __init__(self, layer_sizes, activation_fn):
        super().__init__()
        self.fc = nn.ModuleList()
        if activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation_fn == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        # elif self.activation_fn =="无":
        #     self.activation_fn =None
        for i in range(len(layer_sizes) - 1):
            self.fc.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i != len(layer_sizes) - 2:  # 如果不是倒数第二层(即输出层前一层)，就不加激活函数，防止激活到输出层，损失函数里面已经自带输出层的激活了
                self.fc.append(self.activation_fn)

    def forward(self, x):
        # 遍历ModuleList中的所有层
        for layer in self.fc:
            x = layer(x)  # 循环应用线性层
        return x


class TrainNet():
    """
    主函数，训练并返回结果,训练模型作为生产者线程，结果可视化作为消费者线程，分为两个线程，之前两个步骤相互阻塞导致更新非常不流畅
    """

    def __init__(self, dataset_type="分类", layer_sizes=None, active_fn=None, num_epochs=500, learning_rate=0.1,
                 n_samples=200, batch_size=50, base_n_features=5, selected_derived_features=None, optimizer=None,
                 **kwargs):  # kwargs用于消化参数字典里传入的无关参数(这些参数可能用于别的函数)
        if layer_sizes is None:
            layer_sizes = [200, 64, 32, 1]

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dataset_type = dataset_type
        self.n_samples = n_samples
        self.layer_sizes = layer_sizes
        self.active_fn = active_fn
        self.learning_rate = learning_rate
        self.bach_size = batch_size
        self.base_n_features = base_n_features
        self.selected_derived_features = selected_derived_features
        self.optimizer = optimizer
        # 用于停止训练的信号
        self.stop_event = threading.Event()
        self.model = None

    def train(self):
        try:
            raw_q = queue.Queue(maxsize=1)  # 注意：队列大小设为 1，保证只存“最新快照”
            out_q = queue.Queue(maxsize=1)
            X, y = generate_data(self.dataset_type, self.n_samples, base_n_features=self.base_n_features,
                                 selected_derived_features=self.selected_derived_features)

            def producer():
                if self.stop_event.is_set():  # 检测到停止信号

                    print("训练被手动停止")

                # 2 创建数据，并根据批次加载

                dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).long())
                dataloader = DataLoader(dataset, batch_size=self.bach_size, shuffle=True)

                # 3 构建网络
                self.model = Net(self.layer_sizes, self.active_fn)

                # 4 构建优化器和损失函数
                # 构建优化器
                if self.optimizer == "SGD":
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
                elif self.optimizer == "SGD with Momentum":
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
                elif self.optimizer == "Nesterov":
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9,
                                                nesterov=True)
                elif self.optimizer == "RMSprop":
                    optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
                elif self.optimizer == "Adam":
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

                # 构建损失函数
                criterion = nn.CrossEntropyLoss()
                if self.dataset_type == "回归" or self.dataset_type == "回归2.0":
                    criterion = nn.MSELoss()

                # 5 循环训练
                epochs_list = []
                loss_list = []
                for epoch in range(self.num_epochs):
                    # 传入批次
                    for X_batch, y_batch in dataloader:
                        # 回归与分类的y输入方式不同
                        if self.dataset_type == '回归' or self.dataset_type == '回归2.0':
                            y_batch = y_batch.view(-1, 1)  # [32] -> [32, 1]  # 保持维度一致，MSE的维度要求比较严格
                            y_batch = y_batch.to(torch.float32)

                        optimizer.zero_grad()
                        y_pred = self.model(X_batch)
                        loss = criterion(y_pred, y_batch)
                        loss.backward()
                        # 使用闭包函数更新参数
                        optimizer.step()  # 关键：必须传入闭包函数

                    # 6 每个 epoch 获取权重并绘图,并绘制决策边界
                    weights = []
                    for layer in self.model.fc:
                        if isinstance(layer, nn.Linear):  # 判断类型，防止循环到激活函数
                            weights.append(layer.weight.data.cpu().numpy().T)  # 加上T是因为画图的时候形状不对，加T就好了

                    # 更新epochs_list和loss_list,用于绘制损失曲线
                    epochs_list.append(epoch)
                    loss_list.append(loss.detach().numpy())

                    snapshot = {"epoch": epoch, "loss": loss.item(),
                                "weights": weights,
                                # state_dict用于保存模型的权重和偏置，可以理解为参数快照
                                "state_dict": {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}}
                    if raw_q.full():
                        try:
                            raw_q.get_nowait()  # 丢弃旧的
                        except:
                            pass
                    raw_q.put(snapshot)

                raw_q.put(None)  # 结束信号

            def consumer():
                # 初始化绘图类
                nn_frame = DrawNet(self.layer_sizes)
                draw_effect = DrawEffect(X, y, Net(self.layer_sizes, self.active_fn), dataset_type=self.dataset_type)
                epochs_list, loss_list = [], []

                while True:
                    snap = raw_q.get()
                    if snap is None:
                        break
                    # 1 根据生产者线程返回的参数快照重建模型参数
                    model = draw_effect.model
                    state_dict = {}  # 初始化空字典
                    for k, v in snap["state_dict"].items():
                        # 逐个键值对处理并添加到字典
                        state_dict[k] = torch.from_numpy(v)

                    model.load_state_dict(state_dict, strict=False)

                    # 2 更新 loss 曲线
                    epochs_list.append(snap["epoch"])
                    loss_list.append(snap["loss"])

                    # 绘图
                    nn_frame.ax.clear()
                    nn_frame.draw(snap["weights"])
                    if self.dataset_type.startswith("回归"):
                        draw_effect.draw_fit_surface()
                        draw_effect.draw_loss_curve(epochs_list, loss_list)
                        fig = draw_effect.fit_surface
                    else:
                        draw_effect.draw_decision_boundary()
                        draw_effect.draw_loss_curve(epochs_list, loss_list)
                        fig = draw_effect.boundary_fig

                    if out_q.full():
                        try:
                            out_q.get_nowait()
                        except:
                            pass
                    out_q.put((nn_frame.fig, fig, snap["loss"], snap["weights"], snap["epoch"]))
                # 结束后释放绘图资源
                draw_effect.close_figures()
                out_q.put(None)

            threading.Thread(target=producer, daemon=True).start()
            threading.Thread(target=consumer, daemon=True).start()

            # 主线程 yield
            while True:
                result = out_q.get()
                if result is None:
                    break
                yield result

        except (ValueError, ZeroDivisionError) as e:
            # 获取异常的详细信息
            print(f"捕获异常: {e}")
            # 打印异常发生的位置（文件名、行号等）
            print("异常发生在:")
            traceback.print_exc()  # 打印完整的异常追踪信息
            return e

    #
    def stop_train(self):
        """外部调用此方法，发送停止信号"""
        if not self.stop_event.is_set():
            self.stop_event.set()  # 发送信号（标志设为True）
            print("已发送停止信号")
        else:
            print("已处于停止状态，无需重复发送")

    def save(self,save_path):
        """
        此函数在训练停止或者训练完成后解锁
        :param save_path:
        :return:
        """
        torch.save(self.model,f'{save_path}')  # 保存整个模型
