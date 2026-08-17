# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import get_act_fn
from ..shape_spec import ShapeSpec

__all__ = ['AMFEM']


class SeparableConv(nn.Layer):
    """
    深度可分离卷积 - 轻量化设计
    
    优势:
    - 参数量减少 (相比标准卷积)
    - 计算量减少
    - 保持特征提取能力
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, act='silu'):
        super(SeparableConv, self).__init__()
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        
        # 深度卷积 (Depthwise)
        self.depthwise = nn.Conv2D(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(in_channels)
        
        # 逐点卷积 (Pointwise)
        self.pointwise = nn.Conv2D(in_channels, out_channels, 1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels)
        
        # 激活函数
        if isinstance(act, (str, dict)) or act is None:
            self.act = get_act_fn(act)
        else:
            self.act = act
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class IdentityBlock(nn.Layer):
    """
    残差块 - 特征增强
    
    结构: 1x1 Conv → SepConv → 1x1 Conv + Residual
    """
    def __init__(self, channels, kernel_size=3, dilation=1, act='silu'):
        super(IdentityBlock, self).__init__()
        
        # Bottleneck设计
        bottleneck_channels = channels // 4
        
        self.conv1 = nn.Conv2D(channels, bottleneck_channels, 1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(bottleneck_channels)
        
        self.sep_conv = SeparableConv(bottleneck_channels, bottleneck_channels, 
                                      kernel_size, dilation=dilation, act=act)
        
        self.conv2 = nn.Conv2D(bottleneck_channels, channels, 1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(channels)
        
        if isinstance(act, (str, dict)) or act is None:
            self.act = get_act_fn(act)
        else:
            self.act = act
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        out = self.sep_conv(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + identity
        out = self.act(out)
        
        return out


class SpatialAttention(nn.Layer):
    """
    原始空间注意力 - 单一 7×7 卷积
    
    使用3个统计特征: mean, max, std
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2D(3, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias_attr=False)
        self.bn = nn.BatchNorm2D(1)
        
    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        std_out = paddle.std(x, axis=1, keepdim=True)
        
        x_cat = paddle.concat([avg_out, max_out, std_out], axis=1)
        attention = F.sigmoid(self.bn(self.conv(x_cat)))
        return x * attention


class MultiScaleSpatialAttention(nn.Layer):
    """
    多尺度空间注意力 - 替代原始单一 7×7
    
    改进:
    - 3×3: 精确定位小目标边界
    - 5×5: 中等感受野，平衡细节和上下文
    - 7×7: 大感受野，全局上下文理解
    - 自适应权重: 根据输入特征动态调整各尺度权重
    
    优势:
    - 小目标: 3×3 避免过度平滑
    - 中目标: 5×5 平衡
    - 大目标: 7×7 全局上下文
    - 参数增加: 仅 +118 (可忽略)
    """
    def __init__(self):
        super(MultiScaleSpatialAttention, self).__init__()
        
        # 3×3: 小目标精确定位
        self.conv_3 = nn.Sequential(
            nn.Conv2D(3, 1, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(1)
        )
        
        # 5×5: 中等感受野
        self.conv_5 = nn.Sequential(
            nn.Conv2D(3, 1, kernel_size=5, padding=2, bias_attr=False),
            nn.BatchNorm2D(1)
        )
        
        # 7×7: 大感受野
        self.conv_7 = nn.Sequential(
            nn.Conv2D(3, 1, kernel_size=7, padding=3, bias_attr=False),
            nn.BatchNorm2D(1)
        )
        
        # 自适应权重生成
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(3, 3, 1),
            nn.Softmax(axis=1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W] 注意力加权后的特征
        """
        # 计算统计特征
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        std_out = paddle.std(x, axis=1, keepdim=True)
        stats = paddle.concat([avg_out, max_out, std_out], axis=1)  # [B, 3, H, W]
        
        # 多尺度空间注意力
        attn_3 = self.conv_3(stats)  # [B, 1, H, W]
        attn_5 = self.conv_5(stats)  # [B, 1, H, W]
        attn_7 = self.conv_7(stats)  # [B, 1, H, W]
        
        # 自适应权重
        weights = self.weight_gen(stats)  # [B, 3, 1, 1]
        w3 = weights[:, 0:1, :, :]
        w5 = weights[:, 1:2, :, :]
        w7 = weights[:, 2:3, :, :]
        
        # 加权融合
        attention = w3 * attn_3 + w5 * attn_5 + w7 * attn_7
        attention = F.sigmoid(attention)
        
        return x * attention


class ChannelAttention(nn.Layer):
    """
    增强版通道注意力 - SE机制
    
    使用avg和max双池化，信息更丰富
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        
        self.fc = nn.Sequential(
            nn.Conv2D(channels, channels // reduction, 1, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(channels // reduction, channels, 1, bias_attr=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = F.sigmoid(avg_out + max_out)
        return x * attention


class ECABlock(nn.Layer):
    """
    ECA (Efficient Channel Attention) Block - 双池化增强版
    
    优势:
    - 参数极少: 仅 10~18 个参数 (vs SE的 32K)
    - 计算量低: 无全连接层
    - 双池化: avg捕获全局统计, max捕获显著特征
    - 效果更好: 信息更丰富
    - 局部交互: k个邻居通道交互，更合理
    
    原理:
    - 使用1D卷积代替全连接层
    - 卷积核大小k根据通道数自适应计算
    - 避免降维，保留完整通道信息
    - avg和max双池化融合
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # 自适应计算卷积核大小
        # k = |log₂(C) + b| / γ
        t = int(paddle.abs((paddle.log2(paddle.to_tensor(float(channels))) + b) / gamma).item())
        k_size = t if t % 2 else t + 1  # 确保是奇数
        
        # 双池化
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        
        # 两个1D卷积分别处理avg和max
        self.conv_avg = nn.Conv1D(1, 1, kernel_size=k_size, padding=k_size // 2, bias_attr=False)
        self.conv_max = nn.Conv1D(1, 1, kernel_size=k_size, padding=k_size // 2, bias_attr=False)
        
    def forward(self, x):
        # Avg池化分支: [B, C, H, W] -> [B, C, 1, 1]
        y_avg = self.avg_pool(x)
        # 1D卷积: [B, C, 1, 1] -> [B, 1, C] -> [B, 1, C] -> [B, C, 1, 1]
        y_avg = self.conv_avg(y_avg.squeeze(-1).transpose([0, 2, 1])).transpose([0, 2, 1]).unsqueeze(-1)
        
        # Max池化分支: [B, C, H, W] -> [B, C, 1, 1]
        y_max = self.max_pool(x)
        # 1D卷积: [B, C, 1, 1] -> [B, 1, C] -> [B, 1, C] -> [B, C, 1, 1]
        y_max = self.conv_max(y_max.squeeze(-1).transpose([0, 2, 1])).transpose([0, 2, 1]).unsqueeze(-1)
        
        # 融合两个分支 + Excitation
        y = F.sigmoid(y_avg + y_max)
        
        return x * y.expand_as(x)


class MultiScaleFusion(nn.Layer):
    """
    多尺度特征融合 - MSAM核心
    
    特点:
    - 3个不同尺度的深度可分离卷积 (3x3, 5x5, 7x7)
    - 自适应权重融合 (Softmax归一化)
    - 轻量化设计
    """
    def __init__(self, channels, act='silu'):
        super(MultiScaleFusion, self).__init__()
        
        # ==================== 多尺度分支 ====================
        # 3x3: 小感受野 - 精确定位小目标
        self.branch_3x3 = nn.Sequential(
            nn.Conv2D(channels, channels, 3, padding=1, groups=channels, bias_attr=False),
            nn.BatchNorm2D(channels),
            nn.Conv2D(channels, channels, 1, bias_attr=False),
            nn.BatchNorm2D(channels)
        )
        
        # 5x5: 中等感受野 - 捕获中等目标
        self.branch_5x5 = nn.Sequential(
            nn.Conv2D(channels, channels, 5, padding=2, groups=channels, bias_attr=False),
            nn.BatchNorm2D(channels),
            nn.Conv2D(channels, channels, 1, bias_attr=False),
            nn.BatchNorm2D(channels)
        )
        
        # 7x7: 大感受野 - 全局上下文
        self.branch_7x7 = nn.Sequential(
            nn.Conv2D(channels, channels, 7, padding=3, groups=channels, bias_attr=False),
            nn.BatchNorm2D(channels),
            nn.Conv2D(channels, channels, 1, bias_attr=False),
            nn.BatchNorm2D(channels)
        )
        
        # ==================== 自适应权重生成 ====================
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2D(channels // 4, 3, 1),
            nn.Softmax(axis=1)  # 归一化权重
        )
        
        # 激活函数
        if isinstance(act, (str, dict)) or act is None:
            self.act = get_act_fn(act)
        else:
            self.act = act
            
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] - 多尺度融合后的特征
        """
        # ==================== 多尺度特征提取 ====================
        feat_3 = self.branch_3x3(x)  # [B, C, H, W]
        feat_5 = self.branch_5x5(x)  # [B, C, H, W]
        feat_7 = self.branch_7x7(x)  # [B, C, H, W]
        
        # ==================== 自适应权重计算 ====================
        weights = self.weight_gen(x)  # [B, 3, 1, 1]
        w3 = weights[:, 0:1, :, :]  # [B, 1, 1, 1]
        w5 = weights[:, 1:2, :, :]  # [B, 1, 1, 1]
        w7 = weights[:, 2:3, :, :]  # [B, 1, 1, 1]
        
        # ==================== 自适应加权融合 ====================
        out = feat_3 * w3 + feat_5 * w5 + feat_7 * w7
        out = self.act(out)
        
        return out


class Conv2d_BN(nn.Layer):
    """
    Conv2D + BatchNorm2D 融合模块
    
    特点:
    - 支持任意卷积核大小
    - 支持 BN 权重初始化
    - 支持推理时 Conv+BN 融合（fuse 方法）
    - 轻量化设计
    
    Args:
        a: 输入通道数
        b: 输出通道数
        ks: 卷积核大小（默认1）
        stride: 步长（默认1）
        pad: 填充（默认0）
        dilation: 膨胀率（默认1）
        groups: 分组数（默认1）
        bn_weight_init: BN 权重初始化值（默认1.0）
    """
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1.0):
        super(Conv2d_BN, self).__init__()
        
        # 保存参数用于 fuse
        self.groups = groups
        
        # Conv2D
        self.c = nn.Conv2D(a, b, ks, stride=stride, padding=pad, 
                           dilation=dilation, groups=groups, bias_attr=False)
        
        # BatchNorm2D
        self.bn = nn.BatchNorm2D(b)
        
        # BN 权重和偏置初始化
        init_weight = paddle.full([b], bn_weight_init, dtype='float32')
        init_bias = paddle.zeros([b], dtype='float32')
        self.bn.weight.set_value(init_weight)
        self.bn.bias.set_value(init_bias)
    
    def forward(self, x):
        return self.bn(self.c(x))
    
    @paddle.no_grad()
    def fuse(self):
        """
        融合 Conv + BN 为单个 Conv（推理优化）
        
        原理:
        y = BN(Conv(x))
        y = γ * (Conv(x) - μ) / √(σ² + ε) + β
        y = γ/√(σ² + ε) * Conv(x) + (β - γμ/√(σ² + ε))
        y = W_fused * x + b_fused
        
        其中:
        W_fused = γ/√(σ² + ε) * W_conv
        b_fused = β - γμ/√(σ² + ε)
        
        Returns:
            nn.Conv2D: 融合后的卷积层
        """
        c, bn = self.c, self.bn
        
        # 计算融合权重
        # w = γ / √(σ² + ε)
        w = bn.weight / paddle.sqrt(bn._variance + bn._epsilon)
        
        # 融合卷积权重: W_fused = w * W_conv
        # w 形状: [out_channels] -> [out_channels, 1, 1, 1]
        w_conv = c.weight * w.reshape([-1, 1, 1, 1])
        
        # 融合偏置: b_fused = β - γμ/√(σ² + ε)
        b_conv = bn.bias - bn._mean * w
        
        # 创建融合后的卷积层
        m = nn.Conv2D(
            in_channels=c.weight.shape[1] * self.groups,
            out_channels=c.weight.shape[0],
            kernel_size=c.weight.shape[2:],
            stride=c._stride,
            padding=c._padding,
            dilation=c._dilation,
            groups=self.groups,
            bias_attr=True
        )
        
        # 设置融合后的权重和偏置
        m.weight.set_value(w_conv)
        m.bias.set_value(b_conv)
        
        return m


class FFN(nn.Layer):
    """
    FFN (Feed-Forward Network) - 简化版
    
    结构: Conv2d_BN → Act → Conv2d_BN
    
    对比原始 FFN:
    - 原始: Conv → BN → Act → Dropout → Conv → BN (6层, 分离)
    - 简化: Conv_BN → Act → Conv_BN (4层, 融合)
    
    优势:
    - 结构更简洁
    - 参数量相同
    - 计算量略少（无 Dropout）
    - BN 权重初始化为 0，有助于残差学习
    - 推理时 Conv+BN 可融合，速度更快
    
    Args:
        in_channels: 输入通道数
        expansion: 扩展倍数（默认4）
        dropout: 未使用，保留接口兼容
        act: 激活函数类型（默认 relu）
    """
    def __init__(self, in_channels, expansion=4, dropout=0.0, act='relu'):
        super(FFN, self).__init__()
        hidden_channels = in_channels * expansion
        
        self.pw1 = Conv2d_BN(in_channels, hidden_channels)
        
        # 激活函数
        # if act == 'relu':
        #     self.act = nn.ReLU()
        # elif act == 'silu' or act == 'swish':
        #     self.act = nn.Silu()
        # elif act == 'gelu':
        #     self.act = nn.GELU()
        # else:
        self.act = nn.ReLU()
        
        # BN 权重初始化为 0，有助于残差学习
        self.pw2 = Conv2d_BN(hidden_channels, in_channels, bn_weight_init=0.0)
    
    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class FFN_Original(nn.Layer):
    """
    FFN 原始版本 - 保留用于对比
    
    结构: 1×1 Conv (扩展) → BN → Act → Dropout → 1×1 Conv (压缩) → BN
    """
    def __init__(self, in_channels, expansion=4, dropout=0.0, act='silu'):
        super(FFN_Original, self).__init__()
        hidden_channels = in_channels * expansion
        
        self.fc1 = nn.Conv2D(in_channels, hidden_channels, 1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(hidden_channels)
        self.fc2 = nn.Conv2D(hidden_channels, in_channels, 1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(in_channels)
        self.dropout = nn.Dropout2D(dropout) if dropout > 0 else nn.Identity()
        
        if isinstance(act, (str, dict)) or act is None:
            self.act = get_act_fn(act)
        else:
            self.act = act
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x


@register
@serializable
class AMFEM(nn.Layer):
    """
    AMFEM: Adaptive Multi-scale Feature Enhancement Module
    按照新结构图设计，包含多尺度融合、注意力机制、FFN和多级残差连接
    
    结构流程 (按图):
    C3 → 1×1Conv+BN → 
         [DW 3×3+BN+PW, DW 5×5+BN+PW, DW 7×7+BN+PW, 权重生成] → 加权融合+Silu → 
         通道注意力 → 
         多尺度空间注意力 → 
         FFN + 残差 → 
         残差增强块 → 
         输出投影 → 
         残差连接 → 增强的C3
    
    核心特性:
    1. 多尺度特征融合 (3×3, 5×5, 7×7)
    2. 双池化ECA通道注意力
    3. 多尺度空间注意力
    4. FFN特征变换
    5. 多级残差连接
    
    Args:
        in_channels (int): 输入通道数，ResNet50的C3为512
        out_channels (int): 输出通道数，默认512
        ffn_expansion (int): FFN扩展倍数
        dropout (float): Dropout比例
        act (str): 激活函数类型
        use_eca (bool): 是否使用ECA (True) 或 SE (False)
    """
    
    def __init__(self,
                 in_channels=512,
                 out_channels=512,
                 num_blocks=2,
                 ffn_expansion=4,
                 dropout=0.0,
                 act='silu',
                 use_eca=True,
                 reduction=16,
                 use_ffn=True,
                 num_input_levels=3,
                 stride4_channels=None,
                 c4_channels=1024,
                 c5_channels=2048):
        super(AMFEM, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_ffn = use_ffn
        self.num_input_levels = num_input_levels
        self.stride4_channels = stride4_channels
        self.c4_channels = c4_channels
        self.c5_channels = c5_channels
        
        # ==================== 步骤1: 输入投影 ====================
        self.input_proj = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels)
        )
        
        # ==================== 步骤2: 多尺度特征融合 ====================
        self.multi_scale = MultiScaleFusion(out_channels, act=act)
        
        # ==================== 步骤3: 通道注意力 ====================
        if use_eca:
            self.channel_attn = ECABlock(out_channels)
        else:
            self.channel_attn = ChannelAttention(out_channels, reduction=reduction)
        
        # ==================== 步骤4: 多尺度空间注意力 ====================
        self.spatial_attn = MultiScaleSpatialAttention()
        
        # ==================== 步骤5: FFN ====================
        if use_ffn:
            self.ffn = FFN(out_channels, expansion=ffn_expansion, dropout=dropout, act=act)
        else:
            self.ffn = None
        
        # ==================== 步骤6: 残差增强块 ====================
        self.res_blocks = nn.LayerList([
            IdentityBlock(out_channels, kernel_size=3, act=act)
            for _ in range(num_blocks)
        ])
        
        # ==================== 步骤7: 输出投影 ====================
        self.output_proj = nn.Sequential(
            nn.Conv2D(out_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels)
        )
        
        # 激活函数
        if isinstance(act, (str, dict)) or act is None:
            self.act = get_act_fn(act)
        else:
            self.act = act
            
        # 残差投影
        if in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
                nn.BatchNorm2D(out_channels)
            )
        else:
            self.residual_proj = None
    
    def forward(self, feats):
        """
        前向传播 - 按照结构图实现
        
        流程:
        1. 输入投影 (1×1Conv+BN)
        2. 多尺度融合 (DW 3×3/5×5/7×7 + 权重生成 + 加权融合)
        3. 通道注意力 (ECA双池化)
        4. 多尺度空间注意力 (3×3/5×5/7×7)
        5. FFN + 残差连接
        6. 残差增强块
        7. 输出投影
        8. 全局残差连接
        
        Args:
            feats (list[Tensor]): [C3, C4, C5] 或四尺度 [C2, C3, C4, C5]
        
        Returns:
            list[Tensor]: 与输入长度相同；仅增强 C3
        """
        if len(feats) == 4:
            c2, c3, c4, c5 = feats[0], feats[1], feats[2], feats[3]
        elif len(feats) == 3:
            c2, c3, c4, c5 = None, feats[0], feats[1], feats[2]
        else:
            raise ValueError(
                'AMFEM expects 3 or 4 backbone feature maps, got {}'.format(
                    len(feats)))

        identity = c3

        # ==================== 步骤1: 输入投影 ====================
        x = self.input_proj(c3)  # C3 stride 8
        
        # ==================== 步骤2: 多尺度融合 + 加权融合 + Silu ====================
        x = self.multi_scale(x)  # 内部包含加权融合和激活
        
        # ==================== 步骤3: 通道注意力 ====================
        x = self.channel_attn(x)
        
        # ==================== 步骤4: 多尺度空间注意力 ====================
        x = self.spatial_attn(x)
        
        # ==================== 步骤5: FFN + 残差连接 ====================
        if self.use_ffn:
            x = x + self.ffn(x)  # 残差连接
        
        # ==================== 步骤6: 残差增强块 ====================
        for block in self.res_blocks:
            x = block(x)  # 内部包含残差连接
        
        # ==================== 步骤7: 输出投影 ====================
        x = self.output_proj(x)
        x = self.act(x)
        
        # ==================== 步骤8: 全局残差连接 ====================
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        c3_enhanced = x + identity

        if c2 is not None:
            return [c2, c3_enhanced, c4, c5]
        return [c3_enhanced, c4, c5]
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        n = len(input_shape)
        c3_i = 1 if n == 4 else 0
        d = {
            'in_channels': input_shape[c3_i].channels,
            'num_input_levels': n,
            'c4_channels': input_shape[c3_i + 1].channels,
            'c5_channels': input_shape[c3_i + 2].channels,
        }
        if n == 4:
            d['stride4_channels'] = input_shape[0].channels
        else:
            d['stride4_channels'] = None
        return d
    
    @property
    def out_shape(self):
        if self.num_input_levels == 4:
            return [
                ShapeSpec(
                    channels=self.stride4_channels, stride=4),
                ShapeSpec(channels=self.out_channels, stride=8),
                ShapeSpec(channels=self.c4_channels, stride=16),
                ShapeSpec(channels=self.c5_channels, stride=32),
            ]
        return [
            ShapeSpec(channels=self.out_channels, stride=8),
            ShapeSpec(channels=self.c4_channels, stride=16),
            ShapeSpec(channels=self.c5_channels, stride=32),
        ]


# ==================== 轻量版AMFEM (可选) ====================
@register
@serializable
class AMFEM_Lite(nn.Layer):
    """
    AMFEM轻量版 - 进一步减少计算量
    
    简化:
    - 只使用2个尺度 (3x3, 5x5)
    - 减少残差块数量
    - 简化注意力机制
    
    适合对速度要求极高的场景
    """
    
    def __init__(self,
                 in_channels=512,
                 out_channels=512,
                 act='silu'):
        super(AMFEM_Lite, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 输入投影
        self.input_proj = nn.Conv2D(in_channels, out_channels, 1, bias_attr=False)
        
        # 双尺度分支
        self.branch_3x3 = nn.Conv2D(out_channels, out_channels, 3, padding=1, 
                                     groups=out_channels, bias_attr=False)
        self.branch_5x5 = nn.Conv2D(out_channels, out_channels, 5, padding=2, 
                                     groups=out_channels, bias_attr=False)
        
        # 简化的注意力
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(out_channels, out_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2D(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )
        
        # 激活函数
        if isinstance(act, (str, dict)) or act is None:
            self.act = get_act_fn(act)
        else:
            self.act = act
            
    def forward(self, feats):
        """
        Args:
            feats (list[Tensor]): [C3, C4, C5]
        Returns:
            list[Tensor]: [C3_enhanced, C4, C5]
        """
        c3, c4, c5 = feats[0], feats[1], feats[2]
        
        identity = c3
        
        # 投影
        x = self.input_proj(c3)
        
        # 双尺度特征
        feat_3 = self.branch_3x3(x)
        feat_5 = self.branch_5x5(x)
        x = feat_3 + feat_5
        
        # 注意力
        attn = self.attention(x)
        x = x * attn
        
        # 激活
        x = self.act(x)
        
        # 残差
        if self.in_channels == self.out_channels:
            x = x + identity
        
        return [x, c4, c5]
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': input_shape[0].channels
        }
    
    @property
    def out_shape(self):
        return [
            ShapeSpec(channels=self.out_channels, stride=8),
            ShapeSpec(channels=1024, stride=16),
            ShapeSpec(channels=2048, stride=32)
        ]