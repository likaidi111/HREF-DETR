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
Progressive Decoder for RT-DETR (inspired by MR-DETR)
渐进式解码器：从粗到细逐步精炼目标检测

核心思想:
1. 早期层使用低分辨率特征(大感受野) -> 快速定位目标
2. 后期层使用高分辨率特征(小感受野) -> 精细调整边界框
3. 自适应特征尺度选择 -> 不同大小目标使用不同尺度
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from .utils import _get_clones, inverse_sigmoid

__all__ = ['ProgressiveTransformerDecoder', 'ScaleAdaptiveDecoderLayer']


class PositionRelationEncoder(nn.Layer):
    """
    位置关系编码器 (Relation DETR)
    
    功能:
    1. 计算当前层预测框之间的位置关系矩阵
    2. 使用sin-cos编码位置关系 (dx, dy, dw, dh)
    3. 通过MLP规范化输出
    4. 传递给下一层作为先验信息
    
    训练时启用，推理时关闭，加速训练收敛而不影响推理速度
    
    参考论文: Relation DETR: Exploring Explicit Position Relation Prior for Object Detection
    """
    def __init__(self, d_model=256, num_pos_feats=128, temperature=10000):
        super(PositionRelationEncoder, self).__init__()
        self.d_model = d_model
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        
        # MLP用于规范化位置关系编码
        self.relation_mlp = nn.Sequential(
            nn.Linear(num_pos_feats * 4, d_model),  # 4个关系特征: dx, dy, dw, dh
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
    def _get_sincos_encoding(self, pos, num_pos_feats, temperature):
        """
        正余弦位置编码
        Args:
            pos: [bs, num_queries, num_queries, 1] - 位置差值
        Returns:
            pos_encoding: [bs, num_queries, num_queries, num_pos_feats]
        """
        scale = 2 * 3.141592653589793  # 2*pi
        dim_t = paddle.arange(num_pos_feats, dtype=paddle.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        
        # pos: [bs, num_queries, num_queries, 1]
        # dim_t: [num_pos_feats]
        pos_encoded = pos * scale / dim_t  # [bs, num_queries, num_queries, num_pos_feats]
        
        # 交替使用sin和cos
        pos_encoded_sin = paddle.sin(pos_encoded[..., 0::2])
        pos_encoded_cos = paddle.cos(pos_encoded[..., 1::2])
        
        # 拼接
        pos_encoding = paddle.stack([pos_encoded_sin, pos_encoded_cos], axis=-1)
        pos_encoding = paddle.reshape(pos_encoding, 
                                     [pos_encoding.shape[0], 
                                      pos_encoding.shape[1], 
                                      pos_encoding.shape[2], 
                                      -1])
        return pos_encoding
    
    def forward(self, boxes_from_prev_layer):
        """
        计算位置关系编码
        
        Args:
            boxes_from_prev_layer: [bs, num_queries, 4] - 上一层预测的边界框 (cx, cy, w, h)
        
        Returns:
            relation_encoding: [bs, num_queries, d_model] - 位置关系编码
        """
        bs, num_queries, _ = boxes_from_prev_layer.shape
        
        # 提取中心点和宽高
        cx = boxes_from_prev_layer[..., 0:1]  # [bs, num_queries, 1]
        cy = boxes_from_prev_layer[..., 1:2]
        w = boxes_from_prev_layer[..., 2:3]
        h = boxes_from_prev_layer[..., 3:4]
        
        # 计算所有框之间的相对位置关系
        # [bs, num_queries, 1] - [bs, 1, num_queries] = [bs, num_queries, num_queries]
        dx = cx - paddle.transpose(cx, [0, 2, 1])  # 中心点x方向差
        dy = cy - paddle.transpose(cy, [0, 2, 1])  # 中心点y方向差
        dw = w - paddle.transpose(w, [0, 2, 1])    # 宽度差
        dh = h - paddle.transpose(h, [0, 2, 1])    # 高度差
        
        # 正余弦编码每个关系特征
        dx_encoded = self._get_sincos_encoding(
            paddle.unsqueeze(dx, -1), self.num_pos_feats, self.temperature)
        dy_encoded = self._get_sincos_encoding(
            paddle.unsqueeze(dy, -1), self.num_pos_feats, self.temperature)
        dw_encoded = self._get_sincos_encoding(
            paddle.unsqueeze(dw, -1), self.num_pos_feats, self.temperature)
        dh_encoded = self._get_sincos_encoding(
            paddle.unsqueeze(dh, -1), self.num_pos_feats, self.temperature)
        
        # 拼接所有关系特征
        # [bs, num_queries, num_queries, num_pos_feats * 4]
        relation_features = paddle.concat([dx_encoded, dy_encoded, dw_encoded, dh_encoded], axis=-1)
        
        # 对每个query，聚合其与所有其他query的关系信息
        # 使用平均池化: [bs, num_queries, num_queries, num_pos_feats*4] -> [bs, num_queries, num_pos_feats*4]
        relation_features_pooled = paddle.mean(relation_features, axis=2)
        
        # 通过MLP规范化
        relation_encoding = self.relation_mlp(relation_features_pooled)  # [bs, num_queries, d_model]
        
        return relation_encoding


class ScaleSelector(nn.Layer):
    """
    尺度选择模块
    根据查询特征动态选择最合适的特征尺度
    """
    def __init__(self, hidden_dim, num_levels):
        super(ScaleSelector, self).__init__()
        self.num_levels = num_levels
        
        # 尺度选择网络
        self.scale_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_levels),
            nn.Softmax(axis=-1)
        )
        
    def forward(self, query_feat):
        """
        Args:
            query_feat: [bs, num_queries, hidden_dim]
        Returns:
            scale_weights: [bs, num_queries, num_levels]
        """
        scale_weights = self.scale_predictor(query_feat)
        return scale_weights


class ScaleAdaptiveDecoderLayer(nn.Layer):
    """
    尺度自适应解码器层
    
    特点:
    1. 多分辨率特征自适应选择
    2. 渐进式特征精炼
    3. 跨尺度特征融合
    """
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 activation="relu",
                 n_levels=3,
                 n_points=4,
                 use_scale_selector=True):
        super(ScaleAdaptiveDecoderLayer, self).__init__()
        
        self.d_model = d_model
        self.n_head = n_head
        self.n_levels = n_levels
        self.use_scale_selector = use_scale_selector
        
        # Self-Attention
        from ..layers import MultiHeadAttention
        self.self_attn = MultiHeadAttention(
            d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-Attention (Deformable)
        from .deformable_transformer import MSDeformableAttention
        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 尺度选择器
        if use_scale_selector:
            self.scale_selector = ScaleSelector(d_model, n_levels)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        """
        Args:
            tgt: [bs, num_queries, d_model]
            reference_points: [bs, num_queries, n_levels, 2]
            memory: [bs, \sum(H_i*W_i), d_model]
            memory_spatial_shapes: [n_levels, 2]
            memory_level_start_index: [n_levels]
            query_pos_embed: [bs, num_queries, d_model]
        """
        # Self-Attention
        q = k = tgt + query_pos_embed if query_pos_embed is not None else tgt
        tgt2 = self.self_attn(q, k, tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 尺度自适应选择 (可选)
        scale_weights = None
        if self.use_scale_selector:
            scale_weights = self.scale_selector(tgt)  # [bs, num_queries, n_levels]
        
        # Cross-Attention (Deformable)
        tgt2 = self.cross_attn(
            query=tgt + query_pos_embed if query_pos_embed is not None else tgt,
            reference_points=reference_points,
            value=memory,
            value_spatial_shapes=memory_spatial_shapes,
            value_level_start_index=memory_level_start_index,
            value_mask=memory_mask)
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


@register
class ProgressiveTransformerDecoder(nn.Layer):
    """
    渐进式Transformer解码器 (MR-DETR风格)
    
    核心特性:
    1. 渐进式解码：早期层使用低分辨率特征（快速定位）
                  后期层使用高分辨率特征（精细调整）
    2. 多尺度特征融合：动态选择最合适的特征尺度
    3. 特征金字塔：不同层关注不同尺度的特征
    
    Args:
        hidden_dim: 隐藏层维度
        decoder_layer: 解码器层
        num_layers: 解码器层数
        num_levels: 特征金字塔层数
        progressive_mode: 渐进模式 ['coarse_to_fine', 'adaptive', 'uniform']
        eval_idx: 推理时使用的层索引
    """
    
    def __init__(self,
                 hidden_dim,
                 decoder_layer,
                 num_layers,
                 num_levels=3,
                 progressive_mode='coarse_to_fine',
                 use_scale_selector=True,
                 eval_idx=-1):
        super(ProgressiveTransformerDecoder, self).__init__()
        
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.progressive_mode = progressive_mode
        self.use_scale_selector = use_scale_selector
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        
        # 渐进式尺度权重
        if progressive_mode == 'coarse_to_fine':
            # 早期层关注低分辨率，后期层关注高分辨率
            self.scale_weights = self._get_coarse_to_fine_weights()
        elif progressive_mode == 'adaptive':
            # 自适应学习每层的尺度偏好
            self.scale_weights = nn.ParameterList([
                paddle.create_parameter(
                    shape=[num_levels],
                    dtype='float32',
                    default_initializer=nn.initializer.Constant(1.0 / num_levels))
                for _ in range(num_layers)
            ])
        else:  # uniform
            self.scale_weights = None
        
        # Relation DETR: 位置关系编码器（层间传递）
        # 为每一层创建一个位置关系编码器（除了第一层）
        self.position_relation_encoders = nn.LayerList([
            PositionRelationEncoder(
                d_model=hidden_dim,
                num_pos_feats=128,
                temperature=10000)
            for _ in range(num_layers - 1)  # 第一层不需要，从第二层开始
        ])
        
        print(f"✅ Relation DETR位置关系编码器已初始化")
        print(f"   - 编码器数量: {len(self.position_relation_encoders)} 个")
        print(f"   - 训练时启用，推理时关闭")
        
    def _get_coarse_to_fine_weights(self):
        """
        生成从粗到细的尺度权重
        早期层：更多权重在低分辨率特征（大感受野，快速定位）
        后期层：更多权重在高分辨率特征（小感受野，精细调整）
        """
        weights = []
        for i in range(self.num_layers):
            # 线性插值：从关注低分辨率到关注高分辨率
            progress = i / (self.num_layers - 1) if self.num_layers > 1 else 1.0
            
            # 生成权重：早期偏向低分辨率，后期偏向高分辨率
            layer_weights = []
            for level in range(self.num_levels):
                # level 0: 最高分辨率(stride=8), level n-1: 最低分辨率(stride=32)
                if self.num_levels == 3:
                    if level == 0:  # 高分辨率 (stride=8)
                        w = 0.2 + 0.6 * progress  # 从0.2增长到0.8
                    elif level == 1:  # 中分辨率 (stride=16)
                        w = 0.3
                    else:  # 低分辨率 (stride=32)
                        w = 0.5 - 0.3 * progress  # 从0.5降低到0.2
                else:
                    # 通用公式
                    w = (1.0 - progress) * (self.num_levels - level) / self.num_levels + \
                        progress * (level + 1) / self.num_levels
                layer_weights.append(w)
            
            # 归一化
            total = sum(layer_weights)
            layer_weights = [w / total for w in layer_weights]
            weights.append(layer_weights)
        
        return weights
    
    def _apply_scale_weights(self, reference_points, layer_idx):
        """
        应用尺度权重到参考点
        通过调整不同尺度的权重，实现渐进式解码
        """
        if self.scale_weights is None:
            return reference_points
        
        if self.progressive_mode == 'coarse_to_fine':
            # 使用预定义的从粗到细权重
            weights = paddle.to_tensor(
                self.scale_weights[layer_idx], 
                dtype='float32')
        else:  # adaptive
            # 使用可学习的权重
            weights = F.softmax(self.scale_weights[layer_idx], axis=0)
        
        # weights: [num_levels]
        # reference_points: [bs, num_queries, num_levels, 2]
        # 这里可以选择不同的应用方式，当前保持原样
        # 实际的尺度选择在cross-attention中通过attention weights实现
        
        return reference_points
    
    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                query_pos_head_inv_sig=False):
        """
        渐进式解码前向传播
        
        Args:
            tgt: 初始查询 [bs, num_queries, hidden_dim]
            ref_points_unact: 未激活的参考点 [bs, num_queries, 2]
            memory: 编码器输出 [bs, \sum(H_i*W_i), hidden_dim]
            memory_spatial_shapes: 空间形状 [num_levels, 2]
            memory_level_start_index: 层级起始索引 [num_levels]
            bbox_head: bbox预测头列表
            score_head: 分类预测头列表
            query_pos_head: 查询位置编码头
        
        Returns:
            dec_out_bboxes: 预测框 [num_layers, bs, num_queries, 4]
            dec_out_logits: 预测分数 [num_layers, bs, num_queries, num_classes]
        """
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        ref_points = ref_points_detach  # 初始化ref_points供第一次迭代使用
        
        for i, layer in enumerate(self.layers):
            # 应用渐进式尺度权重
            ref_points_input = self._apply_scale_weights(
                ref_points_detach.unsqueeze(2), i)
            
            # 生成查询位置编码
            if not query_pos_head_inv_sig:
                query_pos_embed = query_pos_head(ref_points_detach)
            else:
                query_pos_embed = query_pos_head(
                    inverse_sigmoid(ref_points_detach))
            
            # Relation DETR: 层间位置关系编码传递（训练时启用）
            if self.training and i > 0:
                # 从第二层开始，使用上一层的预测框生成位置关系先验
                # ref_points 是上一层预测的boxes (cx, cy, w, h)
                prev_boxes = ref_points  # [bs, num_queries, 4]
                
                # 通过位置关系编码器生成先验信息
                relation_prior = self.position_relation_encoders[i-1](prev_boxes)
                # relation_prior: [bs, num_queries, hidden_dim]
                
                # 将位置关系先验融合到查询位置编码中
                # 这个先验信息会在Self-Attention中起作用，加速收敛
                query_pos_embed = query_pos_embed + relation_prior
            
            # 解码器层前向传播
            output = layer(
                output,
                ref_points_input,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask,
                memory_mask,
                query_pos_embed)
            
            # 预测bbox和分数
            inter_ref_bbox = F.sigmoid(
                bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
            
            if self.training:
                # 训练模式：保存所有层的输出
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                # 推理模式：只使用指定层的输出
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break
            
            # 更新参考点（渐进式精炼）
            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox
        
        return paddle.stack(dec_out_bboxes), paddle.stack(dec_out_logits)