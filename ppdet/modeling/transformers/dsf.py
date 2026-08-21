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
DSF (Dynamic Semantic Fusion) module
Fuses FPN features (strong semantics) and PAN features (strong details)

Location: inside HybridEncoder, at FPN–PAN junctions

Core capabilities:
1. Adaptive weighted fusion of FPN and PAN features
2. ECA channel attention
3. Multi-scale spatial attention (improved CBAM)
4. LSKBlock multi-scale adaptation
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
    More efficient than SE; uses 1D convolution instead of FC layers
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # Adaptive computation of convolution kernel size
        t = int(abs((math.log2(float(channels)) + b) / gamma))
        k_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv = nn.Conv1D(1, 1, kernel_size=k_size, padding=k_size // 2, bias_attr=False)
        
    def forward(self, x):
        # Squeeze: [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x)
        # 1D convolution: [B, C, 1, 1] -> [B, 1, C] -> [B, 1, C] -> [B, C, 1, 1]
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


# ==================== Multi-scale spatial attention (improved CBAM) ====================
class MultiScaleSpatialAttention(nn.Layer):
    """
    Multi-scale spatial attention — replaces CBAM’s single 7×7 spatial attention
    
    Original CBAM: only 7×7, may over-smooth small objects
    Improved: 3×3 + 5×5 + 7×7 with adaptive weighted fusion
    
    """
    def __init__(self):
        super(MultiScaleSpatialAttention, self).__init__()
        
        # Input: 3 channels (mean, max, std)
        # Output: 1-channel attention map
        
        # 3×3
        self.conv_3 = nn.Sequential(
            nn.Conv2D(3, 1, kernel_size=3, padding=1, bias_attr=False),
            nn.BatchNorm2D(1)
        )
        
        # 5×5
        self.conv_5 = nn.Sequential(
            nn.Conv2D(3, 1, kernel_size=5, padding=2, bias_attr=False),
            nn.BatchNorm2D(1)
        )
        
        # 7×7
        self.conv_7 = nn.Sequential(
            nn.Conv2D(3, 1, kernel_size=7, padding=3, bias_attr=False),
            nn.BatchNorm2D(1)
        )
        
        # Adaptive Weight Generation
        # Generate weights at 3 scales based on the global statistics of the input features
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),      # [B, 3, H, W] -> [B, 3, 1, 1]
            nn.Conv2D(3, 3, 1),            # [B, 3, 1, 1] -> [B, 3, 1, 1]
            nn.Softmax(axis=1)             # Normalize weights to sum to 1
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] input
        Returns:
            [B, C, H, W] Features weighted by attention
        """
        # ========== Calculate statistical features ==========
        # Calculated along the channel dimension mean, max, std
        avg_out = paddle.mean(x, axis=1, keepdim=True)  # [B, 1, H, W]
        max_out = paddle.max(x, axis=1, keepdim=True)   # [B, 1, H, W]
        std_out = paddle.std(x, axis=1, keepdim=True)   # [B, 1, H, W]
        
        # Concatenated statistical features
        stats = paddle.concat([avg_out, max_out, std_out], axis=1)  # [B, 3, H, W]
        
        # ========== Multi-scale Spatial Attention ==========
        attn_3 = self.conv_3(stats)  # [B, 1, H, W] - Small receptive field
        attn_5 = self.conv_5(stats)  # [B, 1, H, W] - receptive field
        attn_7 = self.conv_7(stats)  # [B, 1, H, W] - big receptive field
        
        # ========== Adaptive Weight Calculation ==========
        weights = self.weight_gen(stats)  # [B, 3, 1, 1]
        w3 = weights[:, 0:1, :, :]  # [B, 1, 1, 1]
        w5 = weights[:, 1:2, :, :]  # [B, 1, 1, 1]
        w7 = weights[:, 2:3, :, :]  # [B, 1, 1, 1]
        
        # ========== Weighted Fusion ==========
        attention = w3 * attn_3 + w5 * attn_5 + w7 * attn_7  # [B, 1, H, W]
        attention = F.sigmoid(attention)
        
        # ========== Apply Attention ==========
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
    LSKBlock

    """
    def __init__(self, channels, act='silu'):
        super(LSKBlock, self).__init__()
        self.channels = channels
        
        # multi-scale branch
        self.dw_conv_3 = nn.Conv2D(channels, channels, 3, padding=1, groups=channels, bias_attr=False)
        self.bn_3 = nn.BatchNorm2D(channels)
        
        self.dw_conv_5 = nn.Conv2D(channels, channels, 5, padding=2, groups=channels, bias_attr=False)
        self.bn_5 = nn.BatchNorm2D(channels)
        
        self.dw_conv_7 = nn.Conv2D(channels, channels, 7, padding=3, groups=channels, bias_attr=False)
        self.bn_7 = nn.BatchNorm2D(channels)
        
        # Space selection
        self.spatial_select = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channels, channels // 4, 1, bias_attr=False),
            nn.BatchNorm2D(channels // 4),
            nn.ReLU()
        )
        
        # Adaptive Weight
        self.attention = nn.Sequential(
            nn.Conv2D(channels // 4, 3, 1, bias_attr=False),
            nn.Softmax(axis=1)
        )
        
        # Feature fusion
        self.fusion = nn.Conv2D(channels, channels, 1, bias_attr=False)
        self.bn_fusion = nn.BatchNorm2D(channels)
        
        if isinstance(act, (str, dict)) or act is None:
            self.act = get_act_fn(act)
        else:
            self.act = act
        
    def forward(self, x):
        identity = x
        
        # multi-scale features
        feat_3 = self.act(self.bn_3(self.dw_conv_3(x)))
        feat_5 = self.act(self.bn_5(self.dw_conv_5(x)))
        feat_7 = self.act(self.bn_7(self.dw_conv_7(x)))
        
        # Adaptive Weight
        spatial_info = self.spatial_select(x)
        weights = self.attention(spatial_info)
        w3 = weights[:, 0:1, :, :]
        w5 = weights[:, 1:2, :, :]
        w7 = weights[:, 2:3, :, :]
        
        # Weighted fusion
        out = feat_3 * w3 + feat_5 * w5 + feat_7 * w7
        out = self.act(self.bn_fusion(self.fusion(out)))
        
        # residual
        return out + identity


# ==================== DSF  ====================
@register
@serializable
class DSF(nn.Layer):
    """
    DSF (Dynamic Semantic Fusion) module
    

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
        
        # ========== Adaptive fusion weight ==========
        self.fusion_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels * 2, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2D(in_channels // 4, 2, 1),
        )
        
        # ========== DW Conv 3×3 ==========
        self.dw_conv = nn.Conv2D(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            groups=in_channels, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(in_channels)
        
        # ========== Channel Attention ==========
        if use_eca:
            self.channel_attn = ECABlock(in_channels)
        else:
            self.channel_attn = SEBlock(in_channels, reduction)
        
        # ========== Spatial attention ==========
        if use_ms_spatial:
            self.spatial_attn = MultiScaleSpatialAttention()
        else:
            # Original CBAM
            self.spatial_attn = nn.Sequential(
                nn.Conv2D(3, 1, kernel_size=7, padding=3, bias_attr=False),
                nn.BatchNorm2D(1),
                nn.Sigmoid()
            )
        
        # ========== FFN ==========
        self.ffn = FFN(
            in_channels,
            hidden_channels=in_channels * ffn_expansion,
            out_channels=in_channels,
            dropout=dropout)
        
        # ========== Final Conv ==========
        self.final_conv = nn.Conv2D(in_channels, in_channels, 1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(in_channels)
        
        # ========== LSKBlock ==========
        self.lsk = LSKBlock(in_channels, act=act)
        
        if isinstance(act, (str, dict)) or act is None:
            self.act = get_act_fn(act)
        else:
            self.act = act

        
    def forward(self, fpn_feat, pan_feat):
        """
        Forward propagation
        
        """
        # ========== Adaptive Fusion ==========
        concat_feat = paddle.concat([fpn_feat, pan_feat], axis=1)
        logits = self.fusion_mlp(concat_feat)
        weights = F.softmax(logits, axis=1)
        w1 = weights[:, 0:1, :, :]
        w2 = weights[:, 1:2, :, :]
        x = fpn_feat * w1 + pan_feat * w2
        
        # ========== DW Conv Residual  ==========
        identity1 = x
        x = self.dw_conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = x + identity1  
        
        # ========== Channel Attention ==========
        x = self.channel_attn(x)
        
        # ========== Spatial attention ==========
        if self.use_ms_spatial:
            # Multi-Scale Spatial Attention (Improved Version)
            x = self.spatial_attn(x)
        else:
            # Original CBAM Spatial Attention
            avg_out = paddle.mean(x, axis=1, keepdim=True)
            max_out = paddle.max(x, axis=1, keepdim=True)
            std_out = paddle.std(x, axis=1, keepdim=True)
            stats = paddle.concat([avg_out, max_out, std_out], axis=1)
            attn = self.spatial_attn(stats)
            x = x * attn
        
        # ========== FFN Residual 2 ==========
        identity2 = x
        x = self.ffn(x)
        x = x + identity2  
        
        # ========== Final Conv ==========
        x = self.final_conv(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # ========== LSKBlock ==========
        x = self.lsk(x)
        return x
