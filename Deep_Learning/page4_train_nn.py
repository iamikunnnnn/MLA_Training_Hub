import queue
import threading
import traceback

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification, make_circles, make_moons, make_regression, make_friedman1
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from Deep_Learning import page4_model_viz, page4_get_data


class Net(nn.Module):
    """
    定义网络
    """

    def __init__(self, layer_sizes, activation_fn,dropout_rate):
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

        self.dropout = nn.Dropout(dropout_rate)
        for i in range(len(layer_sizes) - 1):
            self.fc.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i != len(layer_sizes) - 2:  # 如果不是倒数第二层(即输出层前一层)，则加激活函数，防止激活到输出层，损失函数里面已经自带输出层的激活了
                self.fc.append(self.activation_fn)
                if self.dropout != 0:  #如果丢弃率不为0增加入dropout层
                    self.fc.append(self.dropout)
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
                 n_samples=200, batch_size=8, base_n_features=5, selected_derived_features=None, optimizer=None,
                 scheduler_dict = None,scheduler=None,dropout_rate=None,_lamda=None,**kwargs):  # kwargs用于消化参数字典里传入的无关参数(这些参数可能用于别的函数)
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

        # 学习率调度器
        self.scheduler_dict = scheduler_dict
        self.scheduler = scheduler

        # dropout正则化
        self.dropout_rate = dropout_rate
        # l2正则化
        self._lamda =_lamda

    def train(self):
        try:
            raw_q = queue.Queue(maxsize=1)  # 注意：队列大小设为 1，保证只存“最新快照”
            out_q = queue.Queue(maxsize=1)
            X, y = page4_get_data.generate_data(self.dataset_type, self.n_samples, base_n_features=self.base_n_features,
                                                selected_derived_features=self.selected_derived_features)

            def producer():
                if self.stop_event.is_set():  # 检测到停止信号

                    print("训练被手动停止")

                # 2 创建数据，并根据批次加载

                dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).long())
                dataloader = DataLoader(dataset, batch_size=self.bach_size, shuffle=True)

                # 3 构建网络
                self.model = Net(self.layer_sizes, self.active_fn,self.dropout_rate)

                # 4 构建优化器和损失函数
                # 构建优化器
                if self.optimizer == "SGD":
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,weight_decay=self._lamda)
                elif self.optimizer == "SGD with Momentum":
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9,weight_decay=self._lamda)
                elif self.optimizer == "Nesterov":
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9,weight_decay=self._lamda,nesterov=True)
                elif self.optimizer == "RMSprop":
                    optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,weight_decay=self._lamda)
                elif self.optimizer == "Adam":
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,weight_decay=self._lamda)
                else:
                    return
                # 5 构建损失函数
                criterion = nn.CrossEntropyLoss()
                if self.dataset_type == "回归" or self.dataset_type == "回归2.0":
                    criterion = nn.MSELoss()

                # 6 构建学习率调度器
                if self.scheduler == "StepLR":
                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer=optimizer,
                        step_size=self.scheduler_dict["StepLR"]["step_size"],  # 每十轮更新一次
                        gamma=self.scheduler_dict["StepLR"]["gamma"]  # 衰减率，调整学习率时会*gamma
                    )
                elif self.scheduler =="MultiStepLR":
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer=optimizer,
                        milestones=self.scheduler_dict["MultiStepLR"]["milestones"],
                        gamma=self.scheduler_dict["MultiStepLR"]["gamma"],
                    )
                elif self.scheduler == "ExponentialLR":
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer=optimizer,
                        gamma=self.scheduler_dict["ExponentialLR"]["gamma"],
                    )
                elif self.scheduler == "CosineAnnealingLR":
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=optimizer,
                        T_max=self.scheduler_dict["CosineAnnealingLR"]["T_max"],
                        eta_min=self.scheduler_dict["CosineAnnealingLR"]["eta_min"],
                    )
                elif self.scheduler == "ReduceLROnPlateau":
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer=optimizer,
                        factor=self.scheduler_dict["ReduceLROnPlateau"]["factor"],
                        patience=self.scheduler_dict["ReduceLROnPlateau"]["patience"],
                        min_lr=self.scheduler_dict["ReduceLROnPlateau"]["min_lr"],
                        mode=self.scheduler_dict["ReduceLROnPlateau"]["mode"],
                    )
                elif self.scheduler == "无":
                    pass

                # 7 循环训练
                epochs_list = []
                loss_list = []
                for epoch in range(self.num_epochs):
                    epoch_loss = 0.0  # 用于统计一个 epoch 的平均 loss
                    # 传入批次
                    for X_batch, y_batch in dataloader:
                        # 回归与分类的y输入方式不同
                        if self.dataset_type == '回归' or self.dataset_type == '回归2.0':
                            y_batch = y_batch.view(-1, 1)  # [32] -> [32, 1]  # 保持维度一致，MSE的维度要求比较严格
                            y_batch = y_batch.to(torch.float32)
                        # 清空梯度
                        optimizer.zero_grad()
                        # 前向传播
                        y_pred = self.model(X_batch)
                        # 计算损失
                        loss = criterion(y_pred, y_batch)
                        # 反向传播
                        loss.backward()
                        # 更新参数
                        optimizer.step()

                        # 累加 batch loss
                        epoch_loss += loss.item() * X_batch.size(0)
                    # 计算 epoch 平均 loss
                    epoch_loss /= self.bach_size
                    loss_list.append(epoch_loss)
                    epochs_list.append(epoch)
                    # 训练完成一个 epoch 后再更新学习率调度器
                    if self.scheduler == "无":
                        pass
                    elif self.scheduler == "ReduceLROnPlateau":
                        scheduler.step(metrics=epoch_loss)  # 用 epoch 平均 loss 更新
                    else:
                        scheduler.step()  # 其他调度器按 epoch 更新

                    # 6 每个 epoch 获取权重并绘图,用于绘制决策边界
                    weights = []
                    for layer in self.model.fc:
                        if isinstance(layer, nn.Linear):  # 判断类型，防止循环到激活函数
                            weights.append(layer.weight.data.cpu().numpy().T)  # 加上T是因为画图的时候形状不对，加T就好了

                    # 更新epochs_list和loss_list,用于绘制损失曲线
                    epochs_list.append(epoch)
                    loss_list.append(loss.detach().numpy())

                    # 每n轮更新一次图  # 便于调整epoch显示轮数
                    if epoch % 1 == 0:
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
                nn_frame = page4_model_viz.DrawNet(self.layer_sizes)
                draw_effect = page4_model_viz.DrawEffect(X, y, Net(self.layer_sizes, self.active_fn,self.dropout_rate),
                                                         dataset_type=self.dataset_type)
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

    def save(self, save_path):
        """
        此函数在训练停止或者训练完成后解锁
        :param save_path:
        :return:
        """
        torch.save(self.model, f'{save_path}')  # 保存整个模型
