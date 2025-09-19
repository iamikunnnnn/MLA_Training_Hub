import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import get_dataset  # 保持原有导入
from torchvision import transforms

# LeNet 模型完全不变（无需修改）
class LeNet(nn.Module):
    """还原1998年经典LeNet结构"""

    def __init__(self,
                 conv1_out_channels=6,
                 conv2_out_channels=16,
                 fc1_features=120,
                 fc2_features=84):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.features = nn.Sequential(
            nn.Conv2d(1, conv1_out_channels, kernel_size=5, padding=2),
            self.activation,
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=5),
            self.activation,
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(conv2_out_channels * 5 * 5, fc1_features),
            self.activation,
            nn.Linear(fc1_features, fc2_features),
            self.activation,
            nn.Linear(fc2_features, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self._get_flat_features(x))
        x = self.classifier(x)
        return x

    def _get_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class TrainNet():
    """训练类：仅修改结果返回格式，核心训练逻辑不变"""

    def __init__(self,
                 conv1_out_channels=6,
                 conv2_out_channels=16,
                 fc1_features=120,
                 fc2_features=84,
                 num_epochs=5,
                 batch_size=16):

        # 保留原有参数初始化（无修改）
        self.conv1_out_channels = conv1_out_channels
        self.conv2_out_channels = conv2_out_channels
        self.fc1_features = fc1_features
        self.fc2_features = fc2_features
        self.model = None
        self.num_epochs = num_epochs
        self.batch_size = batch_size


    def train(self):
        """修改点：将模型参数转为可序列化格式"""
        train, test = get_dataset.get_dataset()
        transforms.Lambda(lambda train: torch.to(torch.long))
        dataloader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        self.model = LeNet(
            self.conv1_out_channels,
            self.conv2_out_channels,
            self.fc1_features,
            self.fc2_features
        )
        optimizer = optim.Adam(self.model.parameters())
        loss_fn = nn.CrossEntropyLoss()

        # 训练循环（核心逻辑不变）
        for epoch in range(self.num_epochs):
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()

            # 关键修改：将参数转为列表（PyTorch张量无法直接JSON序列化）
            params = []
            for param in self.model.parameters():
                # 转为numpy数组→列表，保留参数形状信息
                params.append({
                    "shape": param.shape,
                    "values": param.detach().numpy().tolist()  # 转为列表
                })

            # 生成器返回：损失值（转为float）和序列化后的参数
            yield float(loss), params  # 原代码返回 loss（张量）和 self.model.parameters()