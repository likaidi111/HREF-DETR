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
DSF (Deep Semantic Fusion) 模块
用于融合 FPN (语义强) 和 PAN (细节强) 的特征

位置: HybridEncoder 内部，FPN 和 PAN 的连接处

核心功能:
1. 自适应加权融合 FPN 和 PAN 特征
2. ECA 通道注意力
3. 多尺度空间注意力 (改进版CBAM)
4. LSKBlock 多尺度自适应
"""

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import get_act_fn

__all__ = ['DSF']


# ==================== ECA Block ====================
class ECABlock(nn.Layer):
    """
    ECA (Efficient Channel Attention) Block
    比 SE 更高效，使用 1D 卷积代替全连接层
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # 自适应计算卷积核大小
        t = int(abs((math.log2(float(channels)) + b) / gamma))
        k_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv = nn.Conv1D(1, 1, kernel_size=k_size, padding=k_size // 2, bias_attr=False)
        
    def forward(self, x):
        # Squeeze: [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x)
        # 1D卷积: [B, C, 1, 1] -> [B, 1, C] -> [B, 1, C] -> [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose([0, 2, 1])).transpose([0, 2, 1]).unsqueeze(-1)
        # Excitation
        y = F.sigmoid(y)
        return x * y.expand_as(x)


# ==================== SE Block ====================
class SEBlock(nn.Layer):
    """SE (Squeeze-and-Excitation) Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).reshape([b, c])
        y = F.relu(self.fc1(y))
        y = F.sigmoid(self.fc2(y)).reshape([b, c, 1, 1])
        return x * y


# ==================== 多尺度空间注意力 (改进版CBAM) ====================
class MultiScaleSpatialAttention(nn.Layer):
    """
    多尺度空间注意力 - 替代原始 CBAM 的单一 7×7 空间注意力
    
    原始 CBAM: 只用 7×7 卷积，对小目标可能过度平滑
    改进版: 3×3 + 5×5 + 7×7 多尺度，自适应加权融合
    
    优势:
    - 3×3: 精确定位小目标边界
    - 5×5: 中等感受野，平衡细节和上下文
    - 7×7: 大感受野，全局上下文理解
    - 自适应权重: 根据输入特征动态调整各尺度权重
    """
    def __init__(self):
        super(MultiScaleSpatialAttention, self).__init__()
        
        # 输入: 3通道 (mean, max, std)
        # 输出: 1通道 (attention map)
        
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
        
        # 7×7: 大感受野，全局上下文
        self.conv_7 = nn.Sequential(
            nn.Conv2D(3, 1, kernel_size=7, padding=3, bias_attr=False),
            nn.BatchNorm2D(1)
        )
        
        # 自适应权重生成
        # 根据输入特征的全局统计信息生成 3 个尺度的权重
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),      # [B, 3, H, W] -> [B, 3, 1, 1]
            nn.Conv2D(3, 3, 1),            # [B, 3, 1, 1] -> [B, 3, 1, 1]
            nn.Softmax(axis=1)             # 归一化权重，和为1
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 输入特征
        Returns:
            [B, C, H, W] 注意力加权后的特征
        """
        # ========== 步骤1: 计算统计特征 ==========
        # 沿通道维度计算 mean, max, std
        avg_out = paddle.mean(x, axis=1, keepdim=True)  # [B, 1, H, W]
        max_out = paddle.max(x, axis=1, keepdim=True)   # [B, 1, H, W]
        std_out = paddle.std(x, axis=1, keepdim=True)   # [B, 1, H, W]
        
        # 拼接统计特征
        stats = paddle.concat([avg_out, max_out, std_out], axis=1)  # [B, 3, H, W]
        
        # ========== 步骤2: 多尺度空间注意力 ==========
        attn_3 = self.conv_3(stats)  # [B, 1, H, W] - 小感受野
        attn_5 = self.conv_5(stats)  # [B, 1, H, W] - 中感受野
        attn_7 = self.conv_7(stats)  # [B, 1, H, W] - 大感受野
        
        # ========== 步骤3: 自适应权重计算 ==========
        weights = self.weight_gen(stats)  # [B, 3, 1, 1]
        w3 = weights[:, 0:1, :, :]  # [B, 1, 1, 1]
        w5 = weights[:, 1:2, :, :]  # [B, 1, 1, 1]
        w7 = weights[:, 2:3, :, :]  # [B, 1, 1, 1]
        
        # ========== 步骤4: 加权融合 ==========
        attention = w3 * attn_3 + w5 * attn_5 + w7 * attn_7  # [B, 1, H, W]
        attention = F.sigmoid(attention)
        
        # ========== 步骤5: 应用注意力 ==========
        return x * attention


# ==================== FFN ====================
class FFN(nn.Layer):
    """Feed-Forward Network"""
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, dropout=0.0):
        super(FFN, self).__init__()
        hidden_channels = hidden_channels or in_channels * 4
        out_channels = out_channels or in_channels
        
        self.fc1 = nn.Conv2D(in_channels, hidden_channels, 1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(hidden_channels)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2D(hidden_channels, out_channels, 1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels)
        self.dropout = nn.Dropout2D(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x


# ==================== LSKBlock ====================
class LSKBlock(nn.Layer):
    """
    LSKBlock - 大核选择性注意力
    
    多尺度自适应:
    - 3×3: 小目标局部细节
    - 5×5: 中等感受野
    - 7×7: 大感受野，全局上下文
    - 自适应权重: 根据输入动态调整
    """
    def __init__(self, channels, act='silu'):
        super(LSKBlock, self).__init__()
        self.channels = channels
        
        # 多尺度分支
        self.dw_conv_3 = nn.Conv2D(channels, channels, 3, padding=1, groups=channels, bias_attr=False)
        self.bn_3 = nn.BatchNorm2D(channels)
        
        self.dw_conv_5 = nn.Conv2D(channels, channels, 5, padding=2, groups=channels, bias_attr=False)
        self.bn_5 = nn.BatchNorm2D(channels)
        
        self.dw_conv_7 = nn.Conv2D(channels, channels, 7, padding=3, groups=channels, bias_attr=False)
        self.bn_7 = nn.BatchNorm2D(channels)
        
        # 空间选择
        self.spatial_select = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channels, channels // 4, 1, bias_attr=False),
            nn.BatchNorm2D(channels // 4),
            nn.ReLU()
        )
        
        # 自适应权重
        self.attention = nn.Sequential(
            nn.Conv2D(channels // 4, 3, 1, bias_attr=False),
            nn.Softmax(axis=1)
        )
        
        # 特征融合
        self.fusion = nn.Conv2D(channels, channels, 1, bias_attr=False)
        self.bn_fusion = nn.BatchNorm2D(channels)
        
        if isinstance(act, (str, dict)) or act is None:
            self.act = get_act_fn(act)
        else:
            self.act = act
        
    def forward(self, x):
        identity = x
        
        # 多尺度特征
        feat_3 = self.act(self.bn_3(self.dw_conv_3(x)))
        feat_5 = self.act(self.bn_5(self.dw_conv_5(x)))
        feat_7 = self.act(self.bn_7(self.dw_conv_7(x)))
        
        # 自适应权重
        spatial_info = self.spatial_select(x)
        weights = self.attention(spatial_info)
        w3 = weights[:, 0:1, :, :]
        w5 = weights[:, 1:2, :, :]
        w7 = weights[:, 2:3, :, :]
        
        # 加权融合
        out = feat_3 * w3 + feat_5 * w5 + feat_7 * w7
        out = self.act(self.bn_fusion(self.fusion(out)))
        
        # 残差
        return out + identity


# ==================== DSF 主模块 ====================
@register
@serializable
class DSF(nn.Layer):
    """
    DSF (Deep Semantic Fusion) 模块
    
    位置: HybridEncoder 内部，FPN 和 PAN 的连接处
    
    作用: 融合 FPN (语义强) 和 PAN (细节强) 的特征
    
    流程:
    ┌─────────────────────────────────────────────────┐
    │  FPN特征 ──┐                                    │
    │            ├─→ Concat → 权重 → 加权融合          │
    │  PAN特征 ──┘                                    │
    │                    │                            │
    │                    ▼                            │
    │           3×3 DW Conv + 残差                    │
    │                    │                            │
    │                    ▼                            │
    │              ECA 通道注意力                      │
    │                    │                            │
    │                    ▼                            │
    │          多尺度空间注意力 (改进CBAM)             │
    │                    │                            │
    │                    ▼                            │
    │              FFN + 残差                         │
    │                    │                            │
    │                    ▼                            │
    │               1×1 Conv                          │
    │                    │                            │
    │                    ▼                            │
    │         LSKBlock (多尺度自适应)                  │
    │                    │                            │
    │                    ▼                            │
    │                  输出                           │
    └─────────────────────────────────────────────────┘
    
    Args:
        in_channels (int): 输入通道数，默认256
        reduction (int): SE模块的缩减比例
        ffn_expansion (int): FFN的扩展比例
        dropout (float): Dropout比例
        use_eca (bool): 是否使用ECA (True) 或 SE (False)
        use_ms_spatial (bool): 是否使用多尺度空间注意力
        act (str): 激活函数类型
    """
    
    def __init__(self,
                 in_channels=256,
                 reduction=16,
                 ffn_expansion=4,
                 dropout=0.0,
                 use_eca=True,
                 use_ms_spatial=True,
                 act='silu'):
        super(DSF, self).__init__()
        
        self.in_channels = in_channels
        self.use_eca = use_eca
        self.use_ms_spatial = use_ms_spatial
        
        # ========== 1. 自适应融合权重（logits，Softmax 在 forward）==========
        self.fusion_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels * 2, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2D(in_channels // 4, 2, 1),
        )
        
        # ========== 2. DW Conv 3×3 ==========
        self.dw_conv = nn.Conv2D(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            groups=in_channels, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(in_channels)
        
        # ========== 3. 通道注意力 (ECA 或 SE) ==========
        if use_eca:
            self.channel_attn = ECABlock(in_channels)
        else:
            self.channel_attn = SEBlock(in_channels, reduction)
        
        # ========== 4. 空间注意力 (多尺度 或 原始CBAM) ==========
        if use_ms_spatial:
            self.spatial_attn = MultiScaleSpatialAttention()
        else:
            # 原始 CBAM 空间注意力 (单一 7×7)
            self.spatial_attn = nn.Sequential(
                nn.Conv2D(3, 1, kernel_size=7, padding=3, bias_attr=False),
                nn.BatchNorm2D(1),
                nn.Sigmoid()
            )
        
        # ========== 5. FFN ==========
        self.ffn = FFN(
            in_channels,
            hidden_channels=in_channels * ffn_expansion,
            out_channels=in_channels,
            dropout=dropout)
        
        # ========== 6. Final Conv ==========
        self.final_conv = nn.Conv2D(in_channels, in_channels, 1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(in_channels)
        
        # ========== 7. LSKBlock ==========
        self.lsk = LSKBlock(in_channels, act=act)
        
        # 激活函数
        if isinstance(act, (str, dict)) or act is None:
            self.act = get_act_fn(act)
        else:
            self.act = act

        
    def forward(self, fpn_feat, pan_feat):
        """
        前向传播
        
        Args:
            fpn_feat: FPN 上采样后的特征 (语义强)
            pan_feat: PAN 下采样后的特征 (细节强)
            
        Returns:
            融合后的特征
        """
        # ========== 步骤1: 自适应融合 ==========
        concat_feat = paddle.concat([fpn_feat, pan_feat], axis=1)
        logits = self.fusion_mlp(concat_feat)
        weights = F.softmax(logits, axis=1)
        w1 = weights[:, 0:1, :, :]
        w2 = weights[:, 1:2, :, :]
        x = fpn_feat * w1 + pan_feat * w2
        
        # ========== 步骤2: DW Conv + 残差1 ==========
        identity1 = x
        x = self.dw_conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = x + identity1  # 残差连接1
        
        # ========== 步骤3: 通道注意力 (ECA/SE) ==========
        x = self.channel_attn(x)
        
        # ========== 步骤4: 空间注意力 ==========
        if self.use_ms_spatial:
            # 多尺度空间注意力 (改进版)
            x = self.spatial_attn(x)
        else:
            # 原始 CBAM 空间注意力
            avg_out = paddle.mean(x, axis=1, keepdim=True)
            max_out = paddle.max(x, axis=1, keepdim=True)
            std_out = paddle.std(x, axis=1, keepdim=True)
            stats = paddle.concat([avg_out, max_out, std_out], axis=1)
            attn = self.spatial_attn(stats)
            x = x * attn
        
        # ========== 步骤5: FFN + 残差2 ==========
        identity2 = x
        x = self.ffn(x)
        x = x + identity2  # 残差连接2
        
        # ========== 步骤6: Final Conv ==========
        x = self.final_conv(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # ========== 步骤7: LSKBlock ==========
        x = self.lsk(x)
        return x
