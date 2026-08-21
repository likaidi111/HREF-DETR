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
    Depthwise separable convolution — lightweight design
    
    Advantages:
    - Fewer parameters than standard convolution
    - Lower computational cost
    - Preserve feature-extraction capacity
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, act='silu'):
        super(SeparableConv, self).__init__()
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        
        # Depthwise convolution
        self.depthwise = nn.Conv2D(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2D(in_channels, out_channels, 1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels)
        
        # Activation
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
    Residual block — feature enhancement
    
    """
    def __init__(self, channels, kernel_size=3, dilation=1, act='silu'):
        super(IdentityBlock, self).__init__()
        
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
    Original spatial attention
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
    Multi-scale spatial attention — replaces the Original spatial attention
    
    Improvements:
    - 3×3: precise small-object boundary localization
    - 5×5: medium receptive field; balances detail and context
    - 7×7: large receptive field; global context
    - Adaptive weights: dynamically adjust per-scale weights from the input
    
    - Small objects: 3×3 avoids over-smoothing
    - Medium objects: 5×5 for balance
    - Large objects: 7×7 for global context
    - 参Parameter increase: negligible
    """
    def __init__(self):
        super(MultiScaleSpatialAttention, self).__init__()
        
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
        
        # 7×7:
        self.conv_7 = nn.Sequential(
            nn.Conv2D(3, 1, kernel_size=7, padding=3, bias_attr=False),
            nn.BatchNorm2D(1)
        )
        
        # Adaptive weight generation
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
            [B, C, H, W] Adaptive weight generation
        """
        # Compute statistics
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, axis=1, keepdim=True)
        std_out = paddle.std(x, axis=1, keepdim=True)
        stats = paddle.concat([avg_out, max_out, std_out], axis=1)  # [B, 3, H, W]
        
        # Multi-scale spatial attention
        attn_3 = self.conv_3(stats)  # [B, 1, H, W]
        attn_5 = self.conv_5(stats)  # [B, 1, H, W]
        attn_7 = self.conv_7(stats)  # [B, 1, H, W]
        
        # Adaptive weights
        weights = self.weight_gen(stats)  # [B, 3, 1, 1]
        w3 = weights[:, 0:1, :, :]
        w5 = weights[:, 1:2, :, :]
        w7 = weights[:, 2:3, :, :]
        
        # Weighted fusion
        attention = w3 * attn_3 + w5 * attn_5 + w7 * attn_7
        attention = F.sigmoid(attention)
        
        return x * attention


class ChannelAttention(nn.Layer):
    """
    Enhanced channel attention
    
    Uses avg and max dual pooling for richer information
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
    ECA (Efficient Channel Attention) Block with dual-pooling enhancement
    
    Advantage:
    - Very few parameters: only ~10–18 (vs ~32K for SE)
    - Low cost: no fully connected layers
    - Dual pooling: avg for global stats, max for salient features
    - Better performance via richer information
    - Local interaction among k neighboring channels
    
    Principle:
    - Replace FC layers with 1D convolution
    - Kernel size k is computed adaptively from channel count
    - Avoid dimensionality reduction; keep full channel information
    - Uses avg and max dual pooling
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # Adaptively compute kernel size
        # k = |log₂(C) + b| / γ
        t = int(paddle.abs((paddle.log2(paddle.to_tensor(float(channels))) + b) / gamma).item())
        k_size = t if t % 2 else t + 1  # Ensure odd kernel size
        
        # avg and max dual pooling
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.conv_avg = nn.Conv1D(1, 1, kernel_size=k_size, padding=k_size // 2, bias_attr=False)
        self.conv_max = nn.Conv1D(1, 1, kernel_size=k_size, padding=k_size // 2, bias_attr=False)
        
    def forward(self, x):
        # Avg: [B, C, H, W] -> [B, C, 1, 1]
        y_avg = self.avg_pool(x)
        # 1Dconvolution: [B, C, 1, 1] -> [B, 1, C] -> [B, 1, C] -> [B, C, 1, 1]
        y_avg = self.conv_avg(y_avg.squeeze(-1).transpose([0, 2, 1])).transpose([0, 2, 1]).unsqueeze(-1)
        
        # Max: [B, C, H, W] -> [B, C, 1, 1]
        y_max = self.max_pool(x)
        # 1Dconvolution: [B, C, 1, 1] -> [B, 1, C] -> [B, 1, C] -> [B, C, 1, 1]
        y_max = self.conv_max(y_max.squeeze(-1).transpose([0, 2, 1])).transpose([0, 2, 1]).unsqueeze(-1)
        
        # Fuse both branches + excitation
        y = F.sigmoid(y_avg + y_max)
        
        return x * y.expand_as(x)


class MultiScaleFusion(nn.Layer):
    """
    Multi-scale feature fusion
    
    - Three depthwise separable convolutions at different scales (3×3, 5×5, 7×7)
    - Adaptive weight fusion (Softmax normalization)
    - Lightweight design
    """
    def __init__(self, channels, act='silu'):
        super(MultiScaleFusion, self).__init__()
        
        # ==================== MultiScale ====================
        # 3x3
        self.branch_3x3 = nn.Sequential(
            nn.Conv2D(channels, channels, 3, padding=1, groups=channels, bias_attr=False),
            nn.BatchNorm2D(channels),
            nn.Conv2D(channels, channels, 1, bias_attr=False),
            nn.BatchNorm2D(channels)
        )
        
        # 5x5
        self.branch_5x5 = nn.Sequential(
            nn.Conv2D(channels, channels, 5, padding=2, groups=channels, bias_attr=False),
            nn.BatchNorm2D(channels),
            nn.Conv2D(channels, channels, 1, bias_attr=False),
            nn.BatchNorm2D(channels)
        )
        
        # 7x7:
        self.branch_7x7 = nn.Sequential(
            nn.Conv2D(channels, channels, 7, padding=3, groups=channels, bias_attr=False),
            nn.BatchNorm2D(channels),
            nn.Conv2D(channels, channels, 1, bias_attr=False),
            nn.BatchNorm2D(channels)
        )
        
        # ==================== Adaptive weight ====================
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
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W]
        """
        # ==================== Multi-scale feature extraction ====================
        feat_3 = self.branch_3x3(x)  # [B, C, H, W]
        feat_5 = self.branch_5x5(x)  # [B, C, H, W]
        feat_7 = self.branch_7x7(x)  # [B, C, H, W]
        
        # ==================== Adaptive weight calculation ====================
        weights = self.weight_gen(x)  # [B, 3, 1, 1]
        w3 = weights[:, 0:1, :, :]  # [B, 1, 1, 1]
        w5 = weights[:, 1:2, :, :]  # [B, 1, 1, 1]
        w7 = weights[:, 2:3, :, :]  # [B, 1, 1, 1]
        
        # ==================== Adaptive Weighted Fusion ====================
        out = feat_3 * w3 + feat_5 * w5 + feat_7 * w7
        out = self.act(out)
        
        return out


class Conv2d_BN(nn.Layer):
    """
    Conv2D + BatchNorm2D fused module
    
    特点:
    - Supports arbitrary kernel sizes
    - Supports BN weight initialization
    - Supports Conv+BN fusion at inference (fuse)
    - 轻量化设计
    
    Args:
        a: Number of input channels
        b: Number of output channels
        ks: Kernel size (default 1)
        stride: Stride (default 1)
        pad: Padding (default 0)
        dilation: Dilation rate (default 1)
        groups: Number of groups (default 1)
        bn_weight_init: BN weight initialization value (default 1.0)
    """
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1.0):
        super(Conv2d_BN, self).__init__()
        
        self.groups = groups
        
        # Conv2D
        self.c = nn.Conv2D(a, b, ks, stride=stride, padding=pad, 
                           dilation=dilation, groups=groups, bias_attr=False)
        
        # BatchNorm2D
        self.bn = nn.BatchNorm2D(b)
        
        # BN weight and bias initialization
        init_weight = paddle.full([b], bn_weight_init, dtype='float32')
        init_bias = paddle.zeros([b], dtype='float32')
        self.bn.weight.set_value(init_weight)
        self.bn.bias.set_value(init_bias)
    
    def forward(self, x):
        return self.bn(self.c(x))
    
    @paddle.no_grad()
    def fuse(self):
        """
        Fuse Conv + BN into a single Conv (inference optimization)
        
        """
        c, bn = self.c, self.bn
        
        # Calculate fusion weight
        # w = γ / √(σ² + ε)
        w = bn.weight / paddle.sqrt(bn._variance + bn._epsilon)
        
        # Fused convolution weights: W_fused = w * W_conv
        # w: [out_channels] -> [out_channels, 1, 1, 1]
        w_conv = c.weight * w.reshape([-1, 1, 1, 1])
        
        # Fusion Bias: b_fused = β - γμ/√(σ² + ε)
        b_conv = bn.bias - bn._mean * w
        
        # Create a fused convolution layer
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
        
        # Set the fused weights and biases
        m.weight.set_value(w_conv)
        m.bias.set_value(b_conv)
        
        return m


class FFN(nn.Layer):
    """
    FFN (Feed-Forward Network) - Simplified FFN
    
    Structure: Conv2d_BN → Act → Conv2d_BN
    
    Compared :
    - Original: Conv → BN → Act → Dropout → Conv → BN (6 separate layers)
    - Simplified: Conv_BN → Act → Conv_BN (4 fused layers)
    
    """
    def __init__(self, in_channels, expansion=4, dropout=0.0, act='relu'):
        super(FFN, self).__init__()
        hidden_channels = in_channels * expansion
        
        self.pw1 = Conv2d_BN(in_channels, hidden_channels)
        
        # if act == 'relu':
        #     self.act = nn.ReLU()
        # elif act == 'silu' or act == 'swish':
        #     self.act = nn.Silu()
        # elif act == 'gelu':
        #     self.act = nn.GELU()
        # else:
        self.act = nn.ReLU()
        
        # BN weights initialized to 0 to aid residual learning
        self.pw2 = Conv2d_BN(hidden_channels, in_channels, bn_weight_init=0.0)
    
    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class FFN_Original(nn.Layer):
    """
    Original FFN — kept for comparison
    
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
    Includes multi-scale fusion, attention, FFN, and multi-level residuals
    
    Pipeline:
    C3 → 1×1Conv+BN → 
         [DW 3×3+BN+PW, DW 5×5+BN+PW, DW 7×7+BN+PW] → Weighted fusion +Silu → 
         Channel attention / multi-scale spatial attention / FFN + residual / →
         residual enhancement blocks / output projection / residual → enhanced C3
    
    Key properties:
    1. Multi-scale feature fusion (3×3, 5×5, 7×7)
    2. Dual-pooling ECA channel attention
    3. FFN feature transformation
    4. Multi-level residual connections
    
    Args:
        in_channels (int): Input channels; C3 of ResNet50 is 512
        out_channels (int): Output channels, default 512
        ffn_expansion (int): FFN expansion
        dropout (float):  dropout ratio
        act (str): activation type
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
        
        # ==================== input projection  ====================
        self.input_proj = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels)
        )
        
        # ==================== multi-scale fusion ====================
        self.multi_scale = MultiScaleFusion(out_channels, act=act)
        
        # ==================== channel attention ====================
        if use_eca:
            self.channel_attn = ECABlock(out_channels)
        else:
            self.channel_attn = ChannelAttention(out_channels, reduction=reduction)
        
        # ==================== multi-scale spatial attention ====================
        self.spatial_attn = MultiScaleSpatialAttention()
        
        # ==================== FFN ====================
        if use_ffn:
            self.ffn = FFN(out_channels, expansion=ffn_expansion, dropout=dropout, act=act)
        else:
            self.ffn = None
        
        # ==================== residual enhancement ====================
        self.res_blocks = nn.LayerList([
            IdentityBlock(out_channels, kernel_size=3, act=act)
            for _ in range(num_blocks)
        ])
        
        # ==================== output projection ====================
        self.output_proj = nn.Sequential(
            nn.Conv2D(out_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels)
        )
        
        # activation
        if isinstance(act, (str, dict)) or act is None:
            self.act = get_act_fn(act)
        else:
            self.act = act
            
        # residual
        if in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
                nn.BatchNorm2D(out_channels)
            )
        else:
            self.residual_proj = None
    
    def forward(self, feats):
        """
        Forward pass following
        
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

        # ==================== input projection ====================
        x = self.input_proj(c3)  # C3 stride 8
        
        # ==================== multi-scale fusion + Weighted fusion + Silu ====================
        x = self.multi_scale(x)  
        
        # ==================== channel attention ====================
        x = self.channel_attn(x)
        
        # ==================== multi-scale spatial attention ====================
        x = self.spatial_attn(x)
        
        # ==================== FFN + residual ====================
        if self.use_ffn:
            x = x + self.ffn(x) 
        
        # ====================  residual enhancement ====================
        for block in self.res_blocks:
            x = block(x)  
        
        # ==================== output projection ====================
        x = self.output_proj(x)
        x = self.act(x)
        
        # ==================== Global Residual Connection ====================
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


# ==================== Lightweight AMFEM ====================
@register
@serializable
class AMFEM_Lite(nn.Layer):
    """
    Lightweight AMFE: further reduces compute
    
    Simplifications
    - Use only two scales (3×3, 5×5)
    - Fewer residual block
    - Simplified attention
    
    Suitable for highly latency-sensitive settings
    """
    
    def __init__(self,
                 in_channels=512,
                 out_channels=512,
                 act='silu'):
        super(AMFEM_Lite, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.input_proj = nn.Conv2D(in_channels, out_channels, 1, bias_attr=False)
        
        self.branch_3x3 = nn.Conv2D(out_channels, out_channels, 3, padding=1, 
                                     groups=out_channels, bias_attr=False)
        self.branch_5x5 = nn.Conv2D(out_channels, out_channels, 5, padding=2, 
                                     groups=out_channels, bias_attr=False)
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(out_channels, out_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2D(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )
        
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
        

        x = self.input_proj(c3)

        feat_3 = self.branch_3x3(x)
        feat_5 = self.branch_5x5(x)
        x = feat_3 + feat_5
        
        attn = self.attention(x)
        x = x * attn
        
        x = self.act(x)
        
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
