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
from ppdet.modeling.ops import get_act_fn
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = ['LskBlock', 'MKSE']


@register
@serializable
class LskBlock(nn.Layer):
    """
    Enhanced LSKBlock based on the LSNet paper.
    
    Core improvements (Scheme 2):
    1. Adaptive Weight Matrix — dynamically adjusted by the input
    2. Spatial Selection — guided by global information
    3. Large Kernel convolution (7×7) — provides contextual information
    
    Specifically optimized for small-object detection:
    - 3×3: precise small-object localization
    - 5×5: medium receptive field
    - 7×7: global context modeling
    - Adaptive weights: larger w3 for small objects, larger w7 for large objects
    
    Args:
        channels (int): number of input and output channels
        act (str): activation type, default 'silu'
    """
    
    def __init__(self, channels, act="silu"):
        super(LskBlock, self).__init__()
        self.channels = channels
        
        # ==================== Multi-scale branche ====================
        # 3×3 — local details of small objects
        self.dw_conv_3 = nn.Conv2D(
            channels, channels, 3, padding=1,
            groups=channels, bias_attr=False)
        self.bn_3 = nn.BatchNorm2D(channels)
        
        # 5×5 — medium receptive field
        self.dw_conv_5 = nn.Conv2D(
            channels, channels, 5, padding=2,
            groups=channels, bias_attr=False)
        self.bn_5 = nn.BatchNorm2D(channels)
        
        # 7×7 — large receptive field (LSNet core)
        self.dw_conv_7 = nn.Conv2D(
            channels, channels, 7, padding=3,
            groups=channels, bias_attr=False)
        self.bn_7 = nn.BatchNorm2D(channels)
        
        # ==================== Spatial Selection ====================
        # Capture global spatial information
        self.spatial_select = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channels, channels // 4, 1, bias_attr=False),
            nn.BatchNorm2D(channels // 4),
            nn.ReLU()
        )
        
        # ==================== Adaptive weight matrix ====================
        # Generate adaptive weights for the three branches
        self.attention = nn.Sequential(
            nn.Conv2D(channels // 4, 3, 1, bias_attr=False),
            nn.Softmax(axis=1)  # Normalize weights (sum to 1)
        )
        
        # ==================== Feature fusion ====================
        self.fusion = nn.Conv2D(channels, channels, 1, bias_attr=False)
        self.bn_fusion = nn.BatchNorm2D(channels)
        
        # Activation
        if isinstance(act, (str, dict)) or act is None:
            self.act = get_act_fn(act)
        else:
            self.act = act
        
    def forward(self, x):
        """
        Forward pass — LSNet adaptive weight fusion
        
        Pipeline:
        1. Multi-scale feature extraction (3x3, 5x5, 7x7)
        2. Spatial selection
        3. Adaptive weight computation (Softmax normalization)
        4. Weighted fusion
        5. Residual connection
        """
        identity = x
        
        # ====================  Multi-scale feature extraction ====================
        # 3x3
        feat_3 = self.dw_conv_3(x)
        feat_3 = self.bn_3(feat_3)
        feat_3 = self.act(feat_3)
        
        # 5x5
        feat_5 = self.dw_conv_5(x)
        feat_5 = self.bn_5(feat_5)
        feat_5 = self.act(feat_5)
        
        # 7x7
        feat_7 = self.dw_conv_7(x)
        feat_7 = self.bn_7(feat_7)
        feat_7 = self.act(feat_7)
        
        # ==================== Spatial selection ====================
        # Global average pooling captures spatial statistics
        spatial_info = self.spatial_select(x)  # [B, C/4, 1, 1]
        
        # ==================== Adaptive weight matrix ====================
        # Generate normalized weights for the three branches
        weights = self.attention(spatial_info)  # [B, 3, 1, 1]
        
        # Split weights
        w3 = weights[:, 0:1, :, :]  # [B, 1, 1, 1] - 3x3 weights
        w5 = weights[:, 1:2, :, :]  # [B, 1, 1, 1] - 5x5 weights
        w7 = weights[:, 2:3, :, :]  # [B, 1, 1, 1] - 7x7 weights
        
        # ==================== Adaptive weighted fusion ====================
        # Fuse the three branches with adaptive weights
        # Small objects: larger w3 (local details)
        # Large objects: larger w7 (global context)
        out = feat_3 * w3 + feat_5 * w5 + feat_7 * w7
        
        # Feature fusion
        out = self.fusion(out)
        out = self.bn_fusion(out)
        out = self.act(out)
        
        # ==================== Residual connection ====================
        out = out + identity
        
        return out


@register
@serializable
class MKSE(nn.Layer):
    """
    MKSE: Multi-Kernel Shallow Enhancer
    Lightweight C2 (stride=4) enhancement: refine shallow localization features with LskBlock.
    """

    def __init__(self,
                 in_channels,
                 act='silu',
                 num_input_levels=4,
                 c3_channels=512,
                 c4_channels=1024,
                 c5_channels=2048):
        super(MKSE, self).__init__()
        self.in_channels = in_channels
        self.num_input_levels = num_input_levels
        self.c3_channels = c3_channels
        self.c4_channels = c4_channels
        self.c5_channels = c5_channels
        self.lsk = LskBlock(in_channels, act=act)

    def forward(self, feats):
        if len(feats) == 4:
            c2 = self.lsk(feats[0])
            return [c2, feats[1], feats[2], feats[3]]
        if len(feats) == 3:
            return feats
        raise ValueError(
            'MKSE expects 3 or 4 backbone feature maps, got {}'.format(
                len(feats)))

    @classmethod
    def from_config(cls, cfg, input_shape):
        n = len(input_shape)
        d = {
            'in_channels': input_shape[0].channels,
            'num_input_levels': n,
        }
        if n == 4:
            d['c3_channels'] = input_shape[1].channels
            d['c4_channels'] = input_shape[2].channels
            d['c5_channels'] = input_shape[3].channels
        return d

    @property
    def out_shape(self):
        if self.num_input_levels == 4:
            return [
                ShapeSpec(channels=self.in_channels, stride=4),
                ShapeSpec(channels=self.c3_channels, stride=8),
                ShapeSpec(channels=self.c4_channels, stride=16),
                ShapeSpec(channels=self.c5_channels, stride=32),
            ]
        return [
            ShapeSpec(channels=self.c3_channels, stride=8),
            ShapeSpec(channels=self.c4_channels, stride=16),
            ShapeSpec(channels=self.c5_channels, stride=32),
        ]
