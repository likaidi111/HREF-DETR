# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from .meta_arch import BaseArch
from ppdet.core.workspace import register, create

__all__ = ['DETR']
# Deformable DETR, DINO use the same architecture as DETR


@register
class DETR(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process', 'post_process_semi']
    __shared__ = ['with_mask', 'exclude_post_process']

    def __init__(self,
                 backbone,
                 transformer='DETRTransformer',
                 detr_head='DETRHead',
                 neck=None,
                 c2_enhancer=None,
                 c5_enhancer=None,
                 post_process='DETRPostProcess',
                 post_process_semi=None,
                 with_mask=False,
                 exclude_post_process=False):
        super(DETR, self).__init__()
        self.backbone = backbone
        self.c2_enhancer = c2_enhancer
        self.c5_enhancer = c5_enhancer
        self.transformer = transformer
        self.detr_head = detr_head
        self.neck = neck
        self.post_process = post_process
        self.with_mask = with_mask
        self.exclude_post_process = exclude_post_process
        self.post_process_semi = post_process_semi

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}
        c2_enhancer = create(cfg['c2_enhancer'], **kwargs) if cfg.get('c2_enhancer') else None
        if c2_enhancer is not None:
            kwargs = {'input_shape': c2_enhancer.out_shape}
        c5_enhancer = create(cfg['c5_enhancer'], **kwargs) if cfg.get('c5_enhancer') else None
        if c5_enhancer is not None:
            kwargs = {'input_shape': c5_enhancer.out_shape}
        elif c2_enhancer is not None:
            kwargs = {'input_shape': c2_enhancer.out_shape}
        else:
            kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs) if cfg.get('neck') else None
        if neck is not None:
            kwargs = {'input_shape': neck.out_shape}
        elif c5_enhancer is not None:
            kwargs = {'input_shape': c5_enhancer.out_shape}
        elif c2_enhancer is not None:
            kwargs = {'input_shape': c2_enhancer.out_shape}
        else:
            kwargs = {'input_shape': backbone.out_shape}
        transformer = create(cfg['transformer'], **kwargs)
        # head
        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        return {
            'backbone': backbone,
            'c2_enhancer': c2_enhancer,
            'c5_enhancer': c5_enhancer,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck": neck
        }

    def _forward(self):
        # Backbone
        body_feats = self.backbone(self.inputs)
        if self.c2_enhancer is not None:
            body_feats = self.c2_enhancer(body_feats)
        if self.c5_enhancer is not None:
            body_feats = self.c5_enhancer(body_feats)

        # Neck
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        # Transformer
        pad_mask = self.inputs.get('pad_mask', None)
        out_transformer = self.transformer(body_feats, pad_mask, self.inputs)

        # DETR Head
        if self.training:
            detr_losses = self.detr_head(out_transformer, body_feats,
                                         self.inputs)
            detr_losses.update({
                'loss': paddle.add_n(
                    [v for k, v in detr_losses.items() if 'log' not in k])
            })
            return detr_losses
        else:
            preds = self.detr_head(out_transformer, body_feats)
            if self.exclude_post_process:
                bbox, bbox_num, mask = preds
            else:
                bbox, bbox_num, mask = self.post_process(
                    preds, self.inputs['im_shape'], self.inputs['scale_factor'],
                    self.inputs['image'][2:].shape)

            output = {'bbox': bbox, 'bbox_num': bbox_num}
            if self.with_mask:
                output['mask'] = mask
            return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()
