import numpy as np
import matplotlib.pyplot as plt

def forward(class1_points,class2_points, a_fn=None, lr=0.3, Epochs=1000, w=0, b=0):
    # 提取x坐标
    x1 = np.concatenate((class1_points[:, 0], class2_points[:, 0]), axis=0)
    # 提取y坐标
    x2 = np.concatenate((class1_points[:, 1], class2_points[:, 1]), axis=0)
    # 生成标签（class1为0，class2为1）
    labels = np.concatenate((np.zeros(len(class1_points)), np.ones(len(class2_points))), axis=0)
    '2 激活函数'

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    '3 期望函数(用于计算激活值)'

    def f(w1, w2, x1, x2, b):
        z = w1 * x1 + w2 * x2 + b
        a = sigmoid(z)
        return a

    '4 损失函数'

    # %%
    def loss_fn(a, label):
        """
        定义交叉熵损失函数 CrossEntropy Loss
        :param a: 激活值
        :param label: 真实标签
        :return:
        """
        return -np.mean(label * np.log(a) + (1 - label) * np.log(1 - a))

    '5 超参数和参数初始化'

    lr = 0.05
    Epochs = 1000
    w1 = 0
    w2 = 0
    b = 0

    '6 循环训练'
    # 绘图准备
    import matplotlib

    fig, (ax1, ax2) = plt.subplots(1, 2)
    # 用于绘制损失下降图
    epoch_list = []
    loss_list = []

    for epoch in range(Epochs):
        # 1 前向传播
        a = f(w1, w2, x1, x2, b)
        loss_val = loss_fn(a, labels)

        # 2 反向传播 (e(损失)对a求导，a对z求导，z对W求导)，最后获得e对每个w 的偏导数
        # 这里计算的是每个样本的导数
        deda = (labels - a) / (a * (1 - a))
        dadz = a * (1 - a)

        dzdb = 1
        dzdw1 = x1
        dzdw2 = x2

        # 2.1 计算三个参数的样本平均梯度
        gd_w1 = -np.mean(deda * dadz * dzdw1)
        gd_w2 = -np.mean(deda * dadz * dzdw2)
        gd_b = -np.mean(deda * dadz * dzdb)

        # 3 更新参数
        w1 = w1 - lr * gd_w1
        w2 = w2 - lr * gd_w2
        b = b - lr * gd_b

        epoch_list.append(epoch)
        loss_list.append(loss_val)

        # 4 打印训练信息
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"{epoch}/{Epochs}, loss: {loss_val}")
            ax1.clear()
            ax1.scatter(class1_points[:, 0], class1_points[:, 1], color="r")
            ax1.scatter(class2_points[:, 0], class2_points[:, 1], color="b")

            """
            在二分类问题中（例如用 sigmoid 激活的逻辑回归），模型的输出 a 表示样本属于正类的概率。当：
    
            a ≥ 0.5 时，模型预测为正类（标签 1）
            a < 0.5 时，模型预测为负类（标签 0）
    
            而 a 由加权和 z = w1*x1 + w2*x2 + b 通过 sigmoid 函数计算得到。当 a = 0.5 时，对应的 z = 0（因为 sigmoid (0) = 0.5），此时的 x1 和 x2 构成了决策边界，即：
            w1*x1 + w2*x2 + b = 0
            """
            x1_min, x1_max = x1.min(), x1.max()
            '''
            推导：从决策边界公式 w1*x1 + w2*x2 + b = 0 变形得：
            x2 = -(w1*x1 + b) / w2
    
            即当 x1 = x1_min 时，x2 = -(w1*x1_min + b)/w2（对应决策边界的起点 y 坐标）
            当 x1 = x1_max 时，x2 = -(w1*x1_max + b)/w2（对应决策边界的终点 y 坐标）
            '''
            x2_min, x2_max = -(w1 * x1_min + b) / w2, -(w1 * x1_max + b)

            ax1.plot([x1_min, x1_max], [x2_min, x2_max], color="g")

            # 绘制损失下降图
            ax2.clear()
            ax2.plot(epoch_list, loss_list)
            yield fig

    '# 7 评估'
    # -----------准确率-------------
    """
    训练过程中为正类的会被预测为接近1，反之接近0，这样会减少误差，梯度更新的目的也是 这个
    所有这里很自然的，a>决策边界的就是预测为1类，小于预测为0类
    """
    # 计算最后的输出值
    a = f(w1, w2, x1, x2, b)
    # 根据sigmoid 的决策边界，将输出概率>0.5的作为  预测结果为1的值
    y_pre = (a >= 0.5).astype(int)  # a>0.5的作为1，<0.5的很自然就是0
    acc = np.mean(labels == y_pre)
    # -----------f1-score-----------------
    def f1_score(TP, FP, FN):
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    TP = np.sum((labels == 1) & (y_pre == 1))
    FP = np.sum((labels == 0) & (y_pre == 1))
    TN = np.sum((labels == 0) & (y_pre == 0))
    FN = np.sum((labels == 1) & (y_pre == 0))
    f1 = f1_score(TP, FP, FN)
    print(f"f1分数为:{f1:.4f}")

    # ------------------混淆矩阵-----------------------

    confusion_matrix = np.array([
        [TN, FP],
        [FN, TP]
    ])
    fig, ax = plt.subplots()
    cax = ax.imshow(confusion_matrix, cmap="Blues")
    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["预测0", "预测1"])
    ax.set_yticklabels(["真实0", "真实1"])
    return (fig,acc,f1)
