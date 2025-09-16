"""
页面1用到的所有知识点
"""

def basic():
    markdown = r"""
1 神经网络核心概念与数学原理
1.1 神经网络的本质：函数拟合器
神经网络本质是参数化的复杂非线性函数，通过堆叠 "线性变换 + 非线性激活" 层，实现对复杂数据分布的拟合。其核心结构包括：

输入层：接收原始特征，无参数；
隐藏层：通过权重（W）和偏置（b）实现特征变换，激活函数引入非线性；
输出层：输出任务结果，需匹配任务类型（如 softmax 对应多分类）。


1.2 核心参数与数学表达
    """
    return markdown
def progress():
    markdown = r"""
# 2 具体流程
1. 前向传播：神经网络的每一层除了输出层和输入层，都有权重$w_{i,j}$、激活值(or输出值)$a_i$、线性输出值$z_i$、偏置$b_i$、输入值$x_{i,j}$，前向传播需要计算每个神经元的输出并传导为下一层的输入
例如：现在计算输出到下一层第一个神经元的值：$a_{1,j}$= $w_{1,j}*x_{1,j}+b_1$

2. 反向传播：(一定要注意更新$w$时的梯度一定来自于每个样本损失函数的梯度的平均)
根据代价函数$L$，它就像一个指挥中心，目标是计算每一层隐藏层参数的偏导数$\frac{\partial L}{\partial W^{[l]}}$和$\frac{\partial L}{\partial b^{[l]}}$，这些偏导数可理解为该层对损失函数的影响程度。
在反向传播时，从输出层开始计算偏导数：每次向前一层（从第$n$层到第1层），当前层的导数都取决于后一层的导数（$\frac{\partial L}{\partial W^{[l]}} \propto \frac{\partial L}{\partial W^{[l+1]}} \cdot f'(z^{[l]})$）。越靠前的层，链式求导的累积项越多，直到所有层的梯度计算完毕，再按$W^{[l]} = W^{[l]} - \eta \cdot \frac{\partial L}{\partial W^{[l]}}$、$b^{[l]} = b^{[l]} - \eta \cdot \frac{\partial L}{\partial b^{[l]}}$（$\eta$为学习率）更新每一层参数。
        """
    return markdown

def activation():
    markdown = r"""
### 0 为什么一定需要激活函数？
度学习中的 “线性层”（全连接层是典型代表），其核心计算是 线性变换：对输入特征向量x，通过权重矩阵W和偏置b输出z，公式为：
$z = W·x + b$
这里的关键是：线性变换的叠加依然是线性变换。我们可以通过 “堆叠 2 层线性层” 的推导直接验证这一点：
第 1 层线性变换：输入x，输出z₁ = W₁·x + b₁（W₁是第 1 层权重，b₁是第 1 层偏置）；
第 2 层线性变换：将z₁作为输入，输出z₂ = W₂·z₁ + b₂（W₂是第 2 层权重，b₂是第 2 层偏置）；
整体合并：把z₁代入z₂，得到：
$z₂ = W₂·(W₁·x + b₁) + b₂ = (W₂·W₁)·x + (W₂·b₁ + b₂)$
令$W_total = W₂·W₁$（新的权重矩阵）、$b_total = W₂·b₁ + b₂$（新的偏置），则整体公式简化为：
$z₂ = W_total·x + b_total$

这与单层线性层的公式完全一致！换句话说：100 层线性层堆叠的效果，等价于 1 层线性层—— 多层结构的 “深度” 在这里完全失去了意义，因为它没有提升模型的表达能力。

### 1 Sigmoid激活函数拟合

1 函数介绍
sigmoid为激活函数的输出就只能在(0,1)内
损失函数为： $e = \frac{1}{n} \cdot \sum_{i=1}^{n} \left[ y_i - Sigmoid(wx_i + b) \right]^2$

常用隐藏层最后一层到输出层的概率输出

2 函数缺点
Sigmoid激活函数有着如下几种缺点：

1. 梯度消失：Sigmoid函数趋近0和1的时候变化率会变得平坦，从导数图像可以看出，当x值趋向两侧时，其导数趋近于0，在反向传播时，使得神经网络在更新参数时几乎无法学习到低层的特征，从而导致训练变得困难。


2. 不以零为中心：Sigmoid函数的输出范围是0到1之间，它的输出不是以零为中心的，会导致其参数只能同时向同一个方向更新，当有两个参数需要朝相反的方向更新时，该激活函数会使模型的收敛速度大大的降低

3. 计算成本高：Sigmoid激活函数引入了exp()函数，导致其计算成本相对较高，尤其在大规模的深度神经网络中，可能会导致训练速度变慢。

4. 不是稀疏激活：Sigmoid函数的输出范围是连续的，并且不会将输入变为稀疏的激活状态。在某些情况下，稀疏激活可以提高模型的泛化能力和学习效率。

### 2 tanh

相对于Sigmoid函数，优势显而易见：
1.输出以零为中心：tanh函数的输出范围是-1到1之间，其均值为零，因此它是零中心的激活函数。相比于Sigmoid函数，tanh函数能够更好地处理数据的中心化和对称性，有助于提高网络的学习效率。 
2.饱和区域更大：在输入绝对值较大时，tanh函数的斜率较大，这使得它在非线性变换上比Sigmoid函数更加陡峭，有助于提供更强的非线性特性，从而提高了网络的表达能力。 
3.良好的输出范围：tanh函数的输出范围在-1到1之间，相比于Sigmoid函数的0到1之间，输出范围更广，有助于减少数据在网络中传播时的数值不稳定性。
但是缺点也同样明显：
1.容易出现梯度消失问题：虽然相比于Sigmoid函数，tanh函数在非饱和区域的斜率较大，但在输入绝对值较大时，其导数仍然会接近于零，可能导致梯度消失问题。 
2.计算难度同样大。 

 ### 3 Relu
 
ReLU特点：
1.稀疏性：ReLU 函数的导数在输入为负数时为零，这意味着在反向传播过程中，只有激活的神经元会传递梯度，从而促进了稀疏激活的现象，有助于减少过拟合。
2.计算高效：ReLU 函数的计算非常简单，并且在实践中被证明是非常高效的。
3.解决梯度消失问题： ReLU函数在输入大于零时输出其本身值，这使得在反向传播时梯度保持为常数1，避免了梯度消失问题。ReLU函数在深度网络中更容易训练。

它的**劣势**：
- 死亡ReLU问题（Dying ReLU）： 在训练过程中，某些神经元可能会遇到“死亡ReLU”问题，即永远不会被激活。如果某个神经元在训练过程中的权重更新导致在其上的输入始终为负值，那么它的输出将永远为零。这意味着该神经元不会再学习或参与后续训练，导致该神经元“死亡”，从而减少了网络的表达能力。

**死亡relu问题理解**
ReLU函数梯度只可以取两个值，当输入小于0时，梯度为0；当输入大于0时，梯度为1，在反向传播过程中，（w新=w旧-学习率*梯度），如果学习率比较大，一个很大的梯度更新后，经过Relu激活函数，可能会导致ReLU神经元更新后的梯度是负数，进而导致下一轮正向传播过程中ReLU神经元的输入是负数，输出是0，由于ReLU神经元的输出为0，在后续迭代的反向过程中，该处的梯度一直为0，相关参数不再变化，从而导致ReLU神经元的输入始终是负数，输出始终为0。即为“死亡ReLU问题”。

- 输出不是以零为中心： ReLU函数的输出范围是从零开始，因此输出不是以零为中心的。这可能会导致训练过程中的参数更新存在偏差，降低了网络的优化能力。
- 不适合所有问题： 尽管ReLU函数在大多数情况下表现良好，但并不是适合所有问题。对于一些问题，特别是在处理一些包含负值的数据时，ReLU函数可能不够理想，可能会产生不良的结果。

---
**总结**：
1. sigmoid 和 tanh：历史经典与局限性
a.sigmoid 输出压缩到(0, 1)，适合概率输出。致命缺点：梯度消失。目前仅保留在二分类输出层（垃圾邮件分类等）。
b.tanh输出以0为中心点范围是(-1, 1)，收敛速度快于sigmoid。缺点：依旧存在梯度消失问题。

2. relu系列：简单高效的标杆
- - 2.1 基本relu 
f(x)=max(0, x)
优点：计算简单高效，x>0的时候，梯度值为1，彻底解决梯度消失问题
缺点：x<0的时候，梯度值为0，导致神经元死亡
应用：CNN卷积层的标配，AlexNet提出，在ResNet中和残差连接配合使用效果极佳
- - Leaky ReLu
f(x) = max(αx, x) α是小常数
优点：稍微了改进了死亡ReLu问题
变种1：PReLU f(x) = max(αx, x) α是可学习的
变种2：ELU 引入指数

### softmax
**softmax就是把最后输出的[a1,a2,a3,....an]拿出来遍历,然后分别计算属于类别i的具体概率**

1. 公式
    - 对于输入向量 x，softmax函数定义为： **在深度学习中输入的这个x就是隐藏层最后一层输出的所有[a1,a2,a3,....an]**
    - $\operatorname{softmax}(x_j) = \frac{e^{x_j}}{\sum_{i=1}^{n} e^{x_i}}$
    - 其中：
    - i 是当前要计算的向量元素索引
    - j 是求和时遍历向量内部的索引
    - n 是向量的总长度
2. 计算过程
    - 指数化：对向量中每个元素 x_j 计算 e^(x_j)
    - 归一化：将每个指数值除以所有指数值的总和
3. 特性
    - 输出值都在 (0,1) 区间内
    - 所有输出值的和为 1
    - 保持原向量的相对大小关系（大的值对应更大的概率）
4. 例子
    - 输入向量 x = [1, 2, 3]：
    - $e^1 ≈ 2.72, e^2 ≈ 7.39, e^3 ≈ 20.09$
    - 总和 ≈ 30.2
    - 输出 ≈ [0.09, 0.24, 0.67]
    - 这样就把原始数值转换成了概率分布，常用于分类任务的最后一层。
5. softmax 特点总结：
    - 概率分布：Softmax函数将输入转换为概率分布，因此在多分类问题中常用于将模型的原始输出转换为概率值。
    - 连续可导：Softmax函数是连续可导的，这使得它可以与梯度下降等优化算法一起使用进行训练。
    - 指数增长：Softmax函数中的指数运算可能会导致数值稳定性问题，特别是当输入较大时。为了解决这个问题，可以通过减去输入向量总的最大值来进行数值稳定的计算。
    - 梯度计算简单：Softmax函数的导数计算相对简单，可以通过对Softmax函数的定义进行微分得到。

#### softmax例子
核心原因在 softmax 公式里的 **指数函数 $e^{x}$** 和 **归一化分母**，两者共同确保“原始输出 $a_j$ 越大，最终概率越高”：

1. 先看分子 $e^{a_j}$：指数函数的特性是“输入越大，输出增长越快”——比如 $a_j=3$ 时 $e^3≈20.08$，$a_j=5$ 时 $e^5≈148.41$，原始 $a_j$ 差 2，指数后差了7倍多。也就是说，$a_j$ 本身的大小差异会被指数函数 **放大**，让大的 $a_j$ 对应的分子更“突出”。

2. 再看分母 $\sum_{k=1}^n e^{a_k}$：这是所有 $a_k$ 对应指数的总和，本质是“归一化操作”——把所有分子的“绝对大小”转化为“相对占比”。比如两个 $a_j$ 分别是 5 和 3，分子是 148.41 和 20.08，分母是 168.49，算出来概率就是 ≈0.88 和 ≈0.12，大的 $a_j$ 对应的分子在分母中占比更高，最终概率自然更大。

简单说：指数函数先“放大”原始 $a_j$ 的大小差异，归一化再把这种差异转化为“概率占比”，所以原始 $a_j$ 越大，最终概率就越大。


# 4 总结

| 激活函数     | 输出范围       | 是否为零均值 | 梯度消失/爆炸                     | 稀疏性 | 计算复杂度         | 平滑性             | 可学习参数 | 适用场景       |
|--------------|----------------|--------------|-----------------------------------|--------|--------------------|--------------------|------------|----------------|
| Sigmoid      | (0,1)          | 否           | 易消失，不易爆炸                  | 否     | 较高（指数运算）   | 是                 | 否         | 二分类输出层   |
| Tanh         | (-1,1)         | 是           | 易消失，不易爆炸                  | 否     | 较高（指数运算）   | 是                 | 否         | 隐藏层         |
| ReLU         | [0,+∞)         | 否           | 不易消失（正区间），易死亡        | 是     | 低                 | 非平滑（零点处）   | 否         | 隐藏层         |
| Leaky ReLU   | (-∞,+∞)        | 否           | 不易消失，不易爆炸                | 是     | 低                 | 是                 | 否（固定α）| 隐藏层         |
| PReLU        | (-∞,+∞)        | 否           | 不易消失，不易爆炸                | 是     | 低                 | 是                 | 是         | 隐藏层         |
| ELU          | (-∞,+∞)        | 是           | 不易消失，不易爆炸                | 是     | 中等（指数运算）   | 是                 | 否         | 隐藏层         |
| Softmax      | (0,1)且和为1   | 否           | 不易消失，不易爆炸                | 否     | 较高（指数运算）   | 是                 | 否         | 多分类输出层   |


### 4.1 如何选择激活函数
1. 按照网络类型选择
- - CNN：ReLU/Leaky ReLU、Swish
- - RNN/LSTM：Tanh、Leaky ReLU
2. 按照任务类型选择
- - 回归任务：隐藏层用 ReLU/GELU
- - 分类任务：输出层用sigmoid（二分类）、softmax（多分类）
3. 按照模型层的位置
- - 输入层：线性激活（直接传播原始特征）
- - 隐藏层：ReLU（通用）、GELU（自然语言）、Swish（计算机视觉）
- - 输出层：softmax/sigmoid（分类）、线性（回归）、Tanh（生成任务）
            """
    return markdown

def cls_forward_code():
    markdown = """
    ```python
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
        '''
        定义交叉熵损失函数 CrossEntropy Loss
        :param a: 激活值
        :param label: 真实标签
        :return:
        '''
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

            '''
            在二分类问题中（例如用 sigmoid 激活的逻辑回归），模型的输出 a 表示样本属于正类的概率。当：
    
            a ≥ 0.5 时，模型预测为正类（标签 1）
            a < 0.5 时，模型预测为负类（标签 0）
    
            而 a 由加权和 z = w1*x1 + w2*x2 + b 通过 sigmoid 函数计算得到。当 a = 0.5 时，对应的 z = 0（因为 sigmoid (0) = 0.5），此时的 x1 和 x2 构成了决策边界，即：
            w1*x1 + w2*x2 + b = 0
            '''
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
    '''
    训练过程中为正类的会被预测为接近1，反之接近0，这样会减少误差，梯度更新的目的也是 这个
    所有这里很自然的，a>决策边界的就是预测为1类，小于预测为0类
    '''
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
    ```
    """
    return markdown

def reg_forward_code():
    markdown ="""
```python
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import traceback

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
```
    """
    return markdown