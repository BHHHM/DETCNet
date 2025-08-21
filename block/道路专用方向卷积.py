import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Pinwheel_shapedConv(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k, s):
        super().__init__()
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class ChannelAttention(nn.Module):
    """道路提取专用的通道注意力机制，增强重要特征"""

    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class RoadExtractionConv(nn.Module):
    """道路提取专用模块：多尺度方向卷积 + 动态方向权重"""

    def __init__(self, c1, c2, k_list=[3, 5, 7], s=1, reduction=8, road_optimized=True):
        """
        c1: 输入通道数
        c2: 输出通道数
        k_list: 卷积核大小列表，默认为[3,5,7]
        s: 步长
        reduction: 注意力机制的缩减比例
        road_optimized: 是否启用道路专用优化
        """
        super().__init__()
        self.num_scales = len(k_list)
        self.road_optimized = road_optimized

        # 修正：计算每个尺度的通道数，确保可被4整除（因为每个方向特征）
        scale_channels = c2 // self.num_scales
        # 确保每个尺度的通道数可被4整除（因为每个方向特征）
        self.scale_channels = (scale_channels // 4) * 4
        # 调整总输出通道数
        self.adjusted_c2 = self.scale_channels * self.num_scales

        # 1. 多尺度方向卷积分支
        self.scale_convs = nn.ModuleList()
        for k in k_list:
            if road_optimized:
                # 计算降维后的通道数
                mid_channels = max(8, c1 // 2)
                # 创建序列：降维卷积 + Pinwheel卷积
                conv = nn.Sequential(
                    Conv(c1, mid_channels, k=1, s=1),  # 1x1卷积降维
                    Pinwheel_shapedConv(mid_channels, self.scale_channels, k, s)
                )
            else:
                conv = Pinwheel_shapedConv(c1, self.scale_channels, k, s)
            self.scale_convs.append(conv)

        # 2. 动态方向权重机制 (道路专用优化)
        self.attn = nn.Sequential(
            nn.Conv2d(c1, max(4, c1 // reduction), 3, padding=1),  # 保留空间信息
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(max(4, c1 // reduction), 4 * self.num_scales, 1),
            nn.Sigmoid() if road_optimized else nn.Softmax(dim=1)  # 道路优化使用Sigmoid
        )

        # 3. 道路特征融合与增强
        self.fusion = nn.Sequential(
            Conv(self.adjusted_c2, c2, 3, p=1),  # 调整为原始c2输出
            ChannelAttention(c2)  # 通道注意力增强重要特征
        )

        # 4. 道路连续性增强模块 (可选)
        if road_optimized:
            self.context = nn.Sequential(
                nn.AvgPool2d(7, stride=1, padding=3),
                Conv(c2, c2 // 4, 1)
            )
            self.context_fusion = Conv(c2 + c2 // 4, c2, 1)

    def forward(self, x):
        # 1. 并行多尺度特征提取
        scale_features = [conv(x) for conv in self.scale_convs]

        # 2. 计算动态方向权重 [B, 4*num_scales, 1, 1]
        attn_weights = self.attn(x)

        # 3. 应用动态权重 (按尺度和方向)
        weighted_features = []
        for i, feat in enumerate(scale_features):
            # 拆分每个尺度的4个方向特征
            dir_feats = torch.chunk(feat, 4, dim=1)

            # 获取对应的4个权重
            start_idx = i * 4
            weights = attn_weights[:, start_idx:start_idx + 4].unsqueeze(-1).unsqueeze(-1)

            # 加权融合 (使用广播机制)
            weighted = torch.cat([w * f for w, f in zip(weights.split(1, 1), dir_feats)], dim=1)
            weighted_features.append(weighted)

        # 4. 多尺度特征拼接
        concat = torch.cat(weighted_features, dim=1)

        # 5. 道路特征融合与增强
        fused = self.fusion(concat)

        # 6. 道路连续性增强 (可选)
        if self.road_optimized:
            context_feat = self.context(fused)
            fused = torch.cat([fused, context_feat], dim=1)
            fused = self.context_fusion(fused)

        return fused


# 测试代码
if __name__ == "__main__":
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建测试输入张量 (batch_size, channels, height, width)
    x = torch.randn(2, 32, 256, 256).to(device)
    print("输入张量形状:", x.shape)

    # 初始化道路提取模块
    road_conv = RoadExtractionConv(
        c1=32,
        c2=64,  # 原始输出通道数
        k_list=[3, 5, 7],  # 多尺度卷积核
        s=1,
        reduction=8,
        road_optimized=True
    ).to(device)

    # 打印调整后的输出通道信息
    print(f"原始请求输出通道: 64")
    print(f"实际输出通道: {road_conv.adjusted_c2} (因为每个尺度通道必须是4的倍数)")

    # 前向传播测试
    output = road_conv(x)
    print("输出张量形状:", output.shape)

    # 参数数量统计
    total_params = sum(p.numel() for p in road_conv.parameters())
    print(f"总参数量: {total_params / 1e6:.2f}M")

    # 测试动态权重生成
    attn_weights = road_conv.attn(x)
    print(f"注意力权重形状: {attn_weights.shape} (应有4*3=12个权重)")

    # 测试各模块输出
    print("\n各尺度输出形状:")
    for i, conv in enumerate(road_conv.scale_convs):
        feat = conv(x)
        print(f"尺度 {i + 1}: {feat.shape}")