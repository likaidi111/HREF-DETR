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
#


"""
MKSE: Multi-Kernel Shallow Enhancer
Data flow: [C2, C3, C4, C5] → enhance C2 only → [C2', C3, C4, C5]
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.modeling.ops import get_act_fn
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = ['LskBlock', 'MKSE']


class LskBlock(nn.Layer):
    """
    LskBlock: internal MKSE operator (multi-kernel 3×3 / 5×5 / 7×7 + adaptive weighting).
    """

    def __init__(self, channels, act="silu"):
        super(LskBlock, self).__init__()
        self.channels = channels

        # ==================== Multi-scale branches ====================
        self.dw_conv_3 = nn.Conv2D(
            channels, channels, 3, padding=1,
            groups=channels, bias_attr=False)
        self.bn_3 = nn.BatchNorm2D(channels)

        self.dw_conv_5 = nn.Conv2D(
            channels, channels, 5, padding=2,
            groups=channels, bias_attr=False)
        self.bn_5 = nn.BatchNorm2D(channels)

        self.dw_conv_7 = nn.Conv2D(
            channels, channels, 7, padding=3,
            groups=channels, bias_attr=False)
        self.bn_7 = nn.BatchNorm2D(channels)

        # ==================== Spatial selection ====================
        self.spatial_select = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channels, channels // 4, 1, bias_attr=False),
            nn.BatchNorm2D(channels // 4),
            nn.ReLU()
        )

        # ==================== Adaptive weights ====================
        self.attention = nn.Sequential(
            nn.Conv2D(channels // 4, 3, 1, bias_attr=False),
            nn.Softmax(axis=1)
        )

        # ==================== Feature fusion ====================
        self.fusion = nn.Conv2D(channels, channels, 1, bias_attr=False)
        self.bn_fusion = nn.BatchNorm2D(channels)

        if isinstance(act, (str, dict)) or act is None:
            self.act = get_act_fn(act)
        else:
            self.act = act

    def forward(self, x):
        identity = x

        feat_3 = self.act(self.bn_3(self.dw_conv_3(x)))
        feat_5 = self.act(self.bn_5(self.dw_conv_5(x)))
        feat_7 = self.act(self.bn_7(self.dw_conv_7(x)))

        spatial_info = self.spatial_select(x)
        weights = self.attention(spatial_info)
        w3 = weights[:, 0:1, :, :]
        w5 = weights[:, 1:2, :, :]
        w7 = weights[:, 2:3, :, :]

        out = feat_3 * w3 + feat_5 * w5 + feat_7 * w7
        out = self.act(self.bn_fusion(self.fusion(out)))
        out = out + identity
        return out


@register
@serializable
class MKSE(nn.Layer):
    """
    MKSE: Multi-Kernel Shallow Enhancer
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
