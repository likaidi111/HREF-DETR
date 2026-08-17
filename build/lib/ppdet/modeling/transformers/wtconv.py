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

__all__ = ['WTConv', 'WTConvFusion', 'WTConvAdaptive', 'WTConvLite']


class WTConv(nn.Layer):
    """
    WTConv - Wavelet Transform Convolution for Downsampling
    
    使用小波变换进行下采样，保留更多细节信息
    特别适合小目标检测场景
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        wavelet (str): 小波类型，默认'haar'
    """
    def __init__(self, in_channels, out_channels, wavelet='haar'):
        super(WTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wavelet = wavelet
        
        # 如果输入输出通道数不同，添加1x1卷积调整
        if in_channels != out_channels:
            self.adjust = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
                nn.BatchNorm2D(out_channels)
            )
        else:
            self.adjust = None
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, C, H, W]
        
        Returns:
            out: [B, C, H/2, W/2]
        """
        # 小波变换，只使用低频分量LL
        LL, _, _, _ = self.dwt2d(x)
        
        # 调整通道数
        if self.adjust is not None:
            LL = self.adjust(LL)
        
        return LL
    
    def dwt2d(self, x):
        """
        2D离散小波变换 (Haar小波)
        
        Args:
            x: [B, C, H, W]
        
        Returns:
            LL, LH, HL, HH: 各为 [B, C, H/2, W/2]
        """
        # 确保输入尺寸是偶数
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            pad_h = H % 2
            pad_w = W % 2
            x = F.pad(x, [0, pad_w, 0, pad_h], mode='replicate')
        
        # Haar小波变换
        # 分离奇偶行列
        x01 = x[:, :, 0::2, :] / 2.0  # 偶数行
        x02 = x[:, :, 1::2, :] / 2.0  # 奇数行
        
        x1 = x01[:, :, :, 0::2]  # 偶数行偶数列
        x2 = x02[:, :, :, 0::2]  # 奇数行偶数列
        x3 = x01[:, :, :, 1::2]  # 偶数行奇数列
        x4 = x02[:, :, :, 1::2]  # 奇数行奇数列
        
        # Haar小波系数
        LL = x1 + x2 + x3 + x4  # 低频-低频 (近似)
        LH = -x1 - x2 + x3 + x4  # 低频-高频 (水平细节)
        HL = -x1 + x2 - x3 + x4  # 高频-低频 (垂直细节)
        HH = x1 - x2 - x3 + x4   # 高频-高频 (对角细节)
        
        return LL, LH, HL, HH


@register
@serializable
class WTConvFusion(nn.Layer):
    """
    WTConvFusion - Wavelet Transform Convolution with Multi-frequency Fusion
    
    使用小波变换进行下采样，并融合多个频率分量
    保留低频语义信息和高频细节信息
    
    特点:
    1. LL分量: 保留主要语义信息
    2. LH分量: 保留水平边缘细节
    3. HL分量: 保留垂直边缘细节
    4. HH分量: 保留对角线细节
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        act (str): 激活函数类型
        use_se (bool): 是否使用SE注意力模块
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 act='silu',
                 use_se=True):
        super(WTConvFusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_se = use_se
        
        # 小波变换后有4个分量，每个通道数为in_channels
        # 融合层: 4*in_channels -> out_channels
        self.fusion = nn.Sequential(
            nn.Conv2D(in_channels * 4, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels),
            get_act_fn(act) if isinstance(act, (str, dict)) else act
        )
        
        # 可选: SE注意力模块，增强重要频率分量
        if use_se:
            self.se = SELayer(out_channels, reduction=16)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, C, H, W]
        
        Returns:
            out: [B, out_channels, H/2, W/2]
        """
        # 小波变换
        LL, LH, HL, HH = self.dwt2d(x)
        
        # 拼接4个频率分量
        # LL: 低频近似 (主要信息)
        # LH: 水平细节 (水平边缘)
        # HL: 垂直细节 (垂直边缘)
        # HH: 对角细节 (纹理)
        out = paddle.concat([LL, LH, HL, HH], axis=1)  # [B, C*4, H/2, W/2]
        
        # 融合多频信息
        out = self.fusion(out)  # [B, out_channels, H/2, W/2]
        
        # SE注意力
        if self.use_se:
            out = self.se(out)
        
        return out
    
    def dwt2d(self, x):
        """
        2D离散小波变换 (Haar小波)
        
        Haar小波是最简单的小波，计算效率高
        适合实时检测任务
        
        Args:
            x: [B, C, H, W]
        
        Returns:
            LL, LH, HL, HH: 各为 [B, C, H/2, W/2]
        """
        # 确保输入尺寸是偶数
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            # 如果是奇数，进行padding
            pad_h = H % 2
            pad_w = W % 2
            x = F.pad(x, [0, pad_w, 0, pad_h], mode='replicate')
        
        # Haar小波变换
        # 分离奇偶行列
        x01 = x[:, :, 0::2, :] / 2.0  # 偶数行
        x02 = x[:, :, 1::2, :] / 2.0  # 奇数行
        
        x1 = x01[:, :, :, 0::2]  # 偶数行偶数列
        x2 = x02[:, :, :, 0::2]  # 奇数行偶数列
        x3 = x01[:, :, :, 1::2]  # 偶数行奇数列
        x4 = x02[:, :, :, 1::2]  # 奇数行奇数列
        
        # Haar小波系数
        LL = x1 + x2 + x3 + x4  # 低频-低频 (近似)
        LH = -x1 - x2 + x3 + x4  # 低频-高频 (水平细节)
        HL = -x1 + x2 - x3 + x4  # 高频-低频 (垂直细节)
        HH = x1 - x2 - x3 + x4   # 高频-高频 (对角细节)
        
        return LL, LH, HL, HH


class SELayer(nn.Layer):
    """
    SE (Squeeze-and-Excitation) 注意力模块
    用于增强重要的频率分量
    """
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias_attr=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias_attr=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).reshape([B, C])
        y = self.fc(y).reshape([B, C, 1, 1])
        return x * y.expand_as(x)


@register
@serializable  
class WTConvAdaptive(nn.Layer):
    """
    WTConvAdaptive - 自适应小波变换卷积
    
    根据输入特征自适应调整各频率分量的权重
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        act (str): 激活函数类型
    """
    def __init__(self, in_channels, out_channels, act='silu'):
        super(WTConvAdaptive, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 为每个频率分量学习独立的权重
        self.ll_conv = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels)
        )
        self.lh_conv = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels)
        )
        self.hl_conv = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels)
        )
        self.hh_conv = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, 1, bias_attr=False),
            nn.BatchNorm2D(out_channels)
        )
        
        # 自适应权重学习
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels, 4, 1),
            nn.Softmax(axis=1)
        )
        
        self.act = get_act_fn(act) if isinstance(act, (str, dict)) else act
    
    def forward(self, x):
        # 小波变换
        LL, LH, HL, HH = self.dwt2d(x)
        
        # 学习自适应权重
        weights = self.weight_net(x)  # [B, 4, 1, 1]
        w_ll, w_lh, w_hl, w_hh = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3], weights[:, 3:4]
        
        # 处理各频率分量
        ll_out = self.ll_conv(LL) * w_ll
        lh_out = self.lh_conv(LH) * w_lh
        hl_out = self.hl_conv(HL) * w_hl
        hh_out = self.hh_conv(HH) * w_hh
        
        # 融合
        out = ll_out + lh_out + hl_out + hh_out
        out = self.act(out)
        
        return out
    
    def dwt2d(self, x):
        """2D离散小波变换"""
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            pad_h = H % 2
            pad_w = W % 2
            x = F.pad(x, [0, pad_w, 0, pad_h], mode='replicate')
        
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        
        LL = x1 + x2 + x3 + x4
        LH = -x1 - x2 + x3 + x4
        HL = -x1 + x2 - x3 + x4
        HH = x1 - x2 - x3 + x4
        
        return LL, LH, HL, HH


@register
@serializable
class WTConvLite(nn.Layer):
    """
    WTConvLite - 超轻量版小波变换下采样
    
    只融合LL和LH分量，减少50%计算量
    保留主要信息和水平边缘细节
    
    特点:
    1. LL分量: 主要语义信息
    2. LH分量: 水平边缘细节 (对小目标最重要)
    3. 放弃HL/HH: 减少计算量
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        act (str): 激活函数类型
    """
    def __init__(self, in_channels, out_channels, act='silu'):
        super(WTConvLite, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 只融合2个分量: LL + LH
        # 计算量减少50%
        self.fusion_conv = nn.Conv2D(in_channels * 2, out_channels, 1, bias_attr=False)
        self.fusion_bn = nn.BatchNorm2D(out_channels)
        # 修复: 正确处理激活函数
        if isinstance(act, (str, dict)) or act is None:
            self.fusion_act = get_act_fn(act) if act else nn.Identity()
        else:
            self.fusion_act = act
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: [B, C, H, W]
        
        Returns:
            out: [B, out_channels, H/2, W/2]
        """
        # 小波变换
        LL, LH, _, _ = self.dwt2d(x)  # 只使用LL和LH
        
        # 拼接2个频率分量
        # LL: 低频近似 (主要信息)
        # LH: 水平细节 (小目标边缘)
        out = paddle.concat([LL, LH], axis=1)  # [B, C*2, H/2, W/2]
        
        # 融合
        out = self.fusion_conv(out)
        out = self.fusion_bn(out)
        out = self.fusion_act(out)  # [B, out_channels, H/2, W/2]
        
        return out
    
    def dwt2d(self, x):
        """
        2D离散小波变换 (Haar小波)
        
        Args:
            x: [B, C, H, W]
        
        Returns:
            LL, LH, HL, HH: 各为 [B, C, H/2, W/2]
        """
        # 确保输入尺寸是偶数
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            pad_h = H % 2
            pad_w = W % 2
            x = F.pad(x, [0, pad_w, 0, pad_h], mode='replicate')
        
        # Haar小波变换
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        
        LL = x1 + x2 + x3 + x4
        LH = -x1 - x2 + x3 + x4
        HL = -x1 + x2 - x3 + x4
        HH = x1 - x2 - x3 + x4
        
        return LL, LH, HL, HH