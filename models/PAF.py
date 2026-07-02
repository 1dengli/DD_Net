import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 全局平均池化与最大池化：提取通道统计信息
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 通道压缩-激活-恢复：生成通道权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 压缩
        self.relu1 = nn.ReLU()  # 激活
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 恢复
        self.sigmoid = nn.Sigmoid()  # 归一化权重

    def forward(self, x):
        # 平均池化分支
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大池化分支
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 双分支融合+归一化
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), '卷积核尺寸必须为3或7'
        padding = 3if kernel_size == 7 else 1
        # 基于通道统计的空间权重生成
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # 归一化权重

    def forward(self, x):
        # 通道维度平均池化与最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        # 拼接统计特征+卷积生成空间权重
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        x = self.conv1(x_cat)
        return self.sigmoid(x)

class PAF(nn.Module):
    def __init__(self, channels):
        super(PAF, self).__init__()
        # 注意力模块：分别用于融合前和融合后特征增强
        self.ca1 = ChannelAttention(channels * 2)  # 融合后双通道注意力
        self.ca2 = ChannelAttention(channels)     # 精炼后单通道注意力
        self.sa = SpatialAttention()              # 空间注意力（共享）

        self.relu = nn.ReLU(inplace=True)  # 非线性激活

        # 残差分支：保护原始特征，避免梯度消失
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels * 2)
        )  # 融合特征残差
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels)
        )  # 精炼特征残差

        # 核心融合精炼层：压缩通道+特征交互
        self.center_layer = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, stride=1, padding=1),  # 通道压缩为原尺寸
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),  # 特征精炼
            nn.BatchNorm2d(channels)
        )

    def forward(self, S, T):
        """
        前向传播流程：特征拼接→注意力增强→残差融合→通道压缩→二次增强→输出
        """
        # 步骤1：时空特征拼接（通道维度）
        ST = torch.cat((S, T), dim=1)  # [B, 2×channels, H, W]

        # 步骤2：第一次注意力增强（通道+空间）
        attn_ca = self.ca1(ST)  # 通道注意力权重
        attn_sa = self.sa(ST)   # 空间注意力权重
        out1 = attn_ca * attn_sa * ST  # a

        # 步骤3：残差融合（保护原始拼接特征）
        res1 = self.shortcut1(ST)
        out1 += res1  # [B, 2×channels, H, W]

        # 步骤4：通道压缩与特征精炼
        out2 = self.center_layer(out1)  # [B, channels, H, W]

        # 步骤5：第二次注意力增强（通道+空间）
        attn_ca2 = self.ca2(out2)
        attn_sa2 = self.sa(out2)
        out = attn_ca2 * attn_sa2 * out2  # 二次加权

        # 步骤6：残差融合+激活
        res2 = self.shortcut2(out2)
        out += res2
        out = self.relu(out)

        return out


if __name__ == "__main__":
    device = torch.device('cuda:0'if torch.cuda.is_available() else'cpu')

    x = torch.randn(1, 64, 32, 32).to(device)
    t = torch.randn(1, 64, 32, 32).to(device)
    model = PAF(64).to(device)

    y = model(x, t)

    print("微信公众号：十小大的底层视觉工坊")
    print("VX: shixiaodayyds, 备注【即插即用】添加交流群")
    print("知乎、CSDN：十小大")

    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)