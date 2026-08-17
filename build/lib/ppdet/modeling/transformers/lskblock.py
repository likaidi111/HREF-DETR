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

__all__ = ['LskBlock']


@register
@serializable
class LskBlock(nn.Layer):
    """
    LSKBlock Enhanced - 基于LSNet论文的增强版
    
    核心改进 (方案2):
    1. 自适应权重矩阵 (Adaptive Weight Matrix) - 根据输入动态调整
    2. 空间选择机制 (Spatial Selection) - 全局信息引导
    3. 大核卷积 (Large Kernel 7x7) - 提供上下文信息
    
    专门优化小目标检测:
    - 3x3: 精确定位小目标
    - 5x5: 中等感受野
    - 7x7: 全局上下文理解
    - 自适应权重: 小目标时w3大，大目标时w7大
    
    Args:
        channels (int): 输入和输出的通道数
        act (str): 激活函数类型,默认'silu'
    """
    
    def __init__(self, channels, act="silu"):
        super(LskBlock, self).__init__()
        self.channels = channels
        
        # ==================== 多尺度分支 ====================
        # 分支1: 3x3 - 小目标局部细节
        self.dw_conv_3 = nn.Conv2D(
            channels, channels, 3, padding=1,
            groups=channels, bias_attr=False)
        self.bn_3 = nn.BatchNorm2D(channels)
        
        # 分支2: 5x5 - 中等感受野
        self.dw_conv_5 = nn.Conv2D(
            channels, channels, 5, padding=2,
            groups=channels, bias_attr=False)
        self.bn_5 = nn.BatchNorm2D(channels)
        
        # 分支3: 7x7 - 大感受野 (LSNet核心)
        self.dw_conv_7 = nn.Conv2D(
            channels, channels, 7, padding=3,
            groups=channels, bias_attr=False)
        self.bn_7 = nn.BatchNorm2D(channels)
        
        # ==================== 空间选择 (Spatial Selection) ====================
        # 捕获全局空间信息
        self.spatial_select = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(channels, channels // 4, 1, bias_attr=False),
            nn.BatchNorm2D(channels // 4),
            nn.ReLU()
        )
        
        # ==================== 自适应权重矩阵 ====================
        # 为3个分支生成自适应权重
        self.attention = nn.Sequential(
            nn.Conv2D(channels // 4, 3, 1, bias_attr=False),
            nn.Softmax(axis=1)  # 归一化权重 (和为1)
        )
        
        # ==================== 特征融合 ====================
        self.fusion = nn.Conv2D(channels, channels, 1, bias_attr=False)
        self.bn_fusion = nn.BatchNorm2D(channels)
        
        # 激活函数
        if isinstance(act, (str, dict)) or act is None:
            self.act = get_act_fn(act)
        else:
            self.act = act
        
    def forward(self, x):
        """
        前向传播 - LSNet自适应权重融合
        
        流程:
        1. 多尺度特征提取 (3x3, 5x5, 7x7)
        2. 空间选择 (全局池化)
        3. 自适应权重计算 (Softmax归一化)
        4. 加权融合
        5. 残差连接
        """
        identity = x
        
        # ==================== 步骤1: 多尺度特征提取 ====================
        # 3x3分支 - 小目标局部细节
        feat_3 = self.dw_conv_3(x)
        feat_3 = self.bn_3(feat_3)
        feat_3 = self.act(feat_3)
        
        # 5x5分支 - 中等感受野
        feat_5 = self.dw_conv_5(x)
        feat_5 = self.bn_5(feat_5)
        feat_5 = self.act(feat_5)
        
        # 7x7分支 - 大感受野 (LSNet核心)
        feat_7 = self.dw_conv_7(x)
        feat_7 = self.bn_7(feat_7)
        feat_7 = self.act(feat_7)
        
        # ==================== 步骤2: 空间选择 ====================
        # 全局平均池化捕获空间统计信息
        spatial_info = self.spatial_select(x)  # [B, C/4, 1, 1]
        
        # ==================== 步骤3: 自适应权重矩阵 ====================
        # 为3个分支生成归一化权重
        weights = self.attention(spatial_info)  # [B, 3, 1, 1]
        
        # 分离权重
        w3 = weights[:, 0:1, :, :]  # [B, 1, 1, 1] - 3x3权重
        w5 = weights[:, 1:2, :, :]  # [B, 1, 1, 1] - 5x5权重
        w7 = weights[:, 2:3, :, :]  # [B, 1, 1, 1] - 7x7权重
        
        # ==================== 步骤4: 自适应加权融合 ====================
        # 根据自适应权重融合三个分支
        # 小目标: w3权重大 (局部细节)
        # 大目标: w7权重大 (全局上下文)
        out = feat_3 * w3 + feat_5 * w5 + feat_7 * w7
        
        # 特征融合
        out = self.fusion(out)
        out = self.bn_fusion(out)
        out = self.act(out)
        
        # ==================== 步骤5: 残差连接 ====================
        out = out + identity
        
        return out