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

"""
LSFEM-MSA: Large-Scale Feature Enhancement Module with MSA Block Structure
基于 MSA Block 结构的 LSFEM 增强版

结构:
C3 → 1×1投影 → 多尺度融合+残差 → 通道注意力+残差 → FFN+残差 → 
     多尺度空间注意力+残差 → FFN+残差 → 残差增强块 → 输出投影 → 残差连接 → C3'
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import get_act_fn
from ..shape_spec import ShapeSpec

__all__ = ['LSFEM_MSA']


# ==================== 深度可分离卷积 ====================
class SeparableConv(nn.Layer):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, act='silu'):
        super(SeparableConv, self).__init__()
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        
        self.depthwise = nn.Conv2D(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation,
                                   groups=in_channels, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(in_channels)
        self.pointwise = nn.Conv2D(in_channels, out_channels, 1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels)
        
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


# ==================== 多尺度特征融合 ====================
class MultiScaleFusion(nn.Layer):
    """多尺度特征融合 (3×3, 5×5, 7×7)"""
    def __init__(self, channels, act='silu'):
        super(MultiScaleFusion, self).__init__()
        
        # 3×3 分支
        self.branch_3x3 = nn.Sequential(
            nn.Conv2D(channels, channels, 3, padding=1, groups=channels, bias_attr=False),
            nn.BatchNorm2D(channels),
            nn.Conv2D(channels, channels, 1, bias_attr=False),
            nn.BatchNorm2D(channels)
        )
        
        # 5×5 分支
        self.branch_5x5 = nn.Sequential(
            nn.Conv2D(channels, channels, 5, padding=2, groups=channels, bias_attr=False),
            nn.BatchNorm2D(channels),
            nn.Conv2D(channels, channels, 1, bias_attr=False),
            nn.BatchNorm2D(channels)
        )
        
        # 7×7 分支
        self.branch_7x7 = nn.Sequential(
            nn.Conv2D(channels, channels, 7, padding=3, groups=channels, bias_attr=False),
            nn.BatchNorm2D(channels),
            nn.Conv2D(channels, channels, 1, bias_attr=False),
            nn.BatchNorm2D(channels)
        )
        
        # 权重生成
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2D(channels // 4, 3, 1),
            nn.Softmax(axis=1)
        )
        
        if isinstance(act, (str, dict)) or act is None:
            self.act = get_act_fn(act)
        else:
            self.act = act
            
    def forward(self, x):
        feat_3 = self.branch_3x3(x)
        feat_5 = self.branch_5x5(x)
        feat_7 = self.branch_7x7(x)
        
        weights = self.weight_gen(x)
        w3 = weights[:, 0:1, :, :]
        w5 = weights[:, 1:2, :, :]
        w7 = weights[:, 2:3, :, :]
        
        out = feat_3 * w3 + feat_5 * w5 + feat_7 * w7
        out = self.act(out)
        return out


# ==================== ECA 通道注意力 ====================
class ECABlock(nn.Layer):
    """ECA (Efficient Channel Attention)"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        t = int(abs((paddle.log2(paddle.to_tensor(float(channels))) + b) / gamma).item())
        k_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv = nn.Conv1D(1, 1, kernel_size=k_size, padding=k_size // 2, bias_attr=False)
        
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose([0, 2, 1])).transpose([0, 2, 1]).unsqueeze(-1)
        y = F.sigmoid(y)
        return x * y.expand_as(x)


# ==================== FFN ====================
class FFN(nn.Layer):
    """Feed-Forward Network"""
    def __init__(self, in_channels, expansion=4, dropout=0.0, act='silu'):
        super(FFN, self).__init__()
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


# ==================== 多尺度空间注意力 ====================
class MultiScaleSpatialAttention(nn.Layer):
    """多尺度空间注意力 (3×3, 5×5, 7×7)"""
    def __init__(self):
        super(MultiScaleSpatialAttention, self).__init__()
        
        self.conv_3 = nn.Sequential(
            nn.Conv2D(3, 1, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(1)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2D(3, 1, kernel_size=5, padding=2, bias_attr=False),
            nn.BatchNorm2D(1)
        )
        self.conv_7 = nn.Sequential(
            nn.Conv2D(3, 1, kernel_size=7, padding=3, bias_attr=False),
            nn.BatchNorm2D(1)
        )
        
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(3, 3, 1),
            nn.Softmax(axis=1)
        )
        
    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        std_out = paddle.std(x, axis=1, keepdim=True)
        stats = paddle.concat([avg_out, max_out, std_out], axis=1)
        
        attn_3 = self.conv_3(stats)
        attn_5 = self.conv_5(stats)
        attn_7 = self.conv_7(stats)
        
        weights = self.weight_gen(stats)
        w3 = weights[:, 0:1, :, :]
        w5 = weights[:, 1:2, :, :]
        w7 = weights[:, 2:3, :, :]
        
        attention = w3 * attn_3 + w5 * attn_5 + w7 * attn_7
        attention = F.sigmoid(attention)
        
        return x * attention


# ==================== 残差增强块 ====================
class ResidualBlock(nn.Layer):
    """残差增强块"""
    def __init__(self, channels, act='silu'):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2D(channels, channels, 3, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(channels)
        self.conv2 = nn.Conv2D(channels, channels, 3, padding=1, bias_attr=False)
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
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.act(out)
        return out


# ==================== LSFEM-MSA 主模块 ====================
@register
@serializable
class LSFEM_MSA(nn.Layer):
    """
    LSFEM-MSA: 基于 MSA Block 结构的增强版
    
    结构 (7个残差连接):
    1. 输入投影
    2. 多尺度融合 + 残差
    3. 通道注意力 + 残差
    4. FFN + 残差
    5. 多尺度空间注意力 + 残差
    6. FFN + 残差
    7. 残差增强块
    8. 输出投影 + 残差
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        ffn_expansion (int): FFN 扩展倍数
        dropout (float): Dropout 比例
        act (str): 激活函数
    """
    
    def __init__(self,
                 in_channels=512,
                 out_channels=512,
                 ffn_expansion=4,
                 dropout=0.0,
                 act='silu'):
        super(LSFEM_MSA, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # ==================== 输入投影 ====================
        self.input_proj = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels)
        )
        
        # ==================== 多尺度融合 ====================
        self.multi_scale = MultiScaleFusion(out_channels, act=act)
        
        # ==================== 通道注意力 ====================
        self.channel_attn = ECABlock(out_channels)
        
        # ==================== FFN 1 ====================
        self.ffn1 = FFN(out_channels, expansion=ffn_expansion, dropout=dropout, act=act)
        
        # ==================== 多尺度空间注意力 ====================
        self.spatial_attn = MultiScaleSpatialAttention()
        
        # ==================== FFN 2 ====================
        self.ffn2 = FFN(out_channels, expansion=ffn_expansion, dropout=dropout, act=act)
        
        # ==================== 残差增强块 ====================
        self.res_block = ResidualBlock(out_channels, act=act)
        
        # ==================== 输出投影 ====================
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
        前向传播 - MSA Block 结构
        
        流程:
        1. 输入投影
        2. 多尺度融合 + 残差
        3. 通道注意力 + 残差
        4. FFN + 残差
        5. 多尺度空间注意力 + 残差
        6. FFN + 残差
        7. 残差增强块
        8. 输出投影 + 残差
        
        Args:
            feats (list[Tensor]): [C3, C4, C5]
        
        Returns:
            list[Tensor]: [C3_enhanced, C4, C5]
        """
        c3, c4, c5 = feats[0], feats[1], feats[2]
        identity = c3
        
        # ==================== 步骤1: 输入投影 ====================
        x = self.input_proj(c3)
        
        # ==================== 步骤2: 多尺度融合 + 残差 ====================
        x = x + self.multi_scale(x)
        
        # ==================== 步骤3: 通道注意力 + 残差 ====================
        x = x + self.channel_attn(x)
        
        # ==================== 步骤4: FFN + 残差 ====================
        x = x + self.ffn1(x)
        
        # ==================== 步骤5: 多尺度空间注意力 + 残差 ====================
        x = x + self.spatial_attn(x)
        
        # ==================== 步骤6: FFN + 残差 ====================
        x = x + self.ffn2(x)
        
        # ==================== 步骤7: 残差增强块 ====================
        x = self.res_block(x)
        
        # ==================== 步骤8: 输出投影 + 残差 ====================
        x = self.output_proj(x)
        x = self.act(x)
        
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        c3_enhanced = x + identity
        
        return [c3_enhanced, c4, c5]
    
    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': input_shape[0].channels}
    
    @property
    def out_shape(self):
        return [
            ShapeSpec(channels=self.out_channels, stride=8),
            ShapeSpec(channels=1024, stride=16),
            ShapeSpec(channels=2048, stride=32)
        ]
