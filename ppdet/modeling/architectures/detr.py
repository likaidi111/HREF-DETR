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
from ppdet.core.workspace import register, create, global_config

__all__ = ['DETR']
# Deformable DETR, DINO use the same architecture as DETR


def _apply_scale_config(use_4_scale):
    """Sync ResNet / HybridEncoder / RTDETRTransformer for 3 or 4 scales."""
    if use_4_scale:
        return_idx = [0, 1, 2, 3]
        use_encoder_idx = [3]
        feat_strides = [4, 8, 16, 32]
        num_levels = 4
    else:
        return_idx = [1, 2, 3]
        use_encoder_idx = [2]
        feat_strides = [8, 16, 32]
        num_levels = 3

    if 'ResNet' in global_config:
        global_config['ResNet']['return_idx'] = list(return_idx)
    if 'HybridEncoder' in global_config:
        global_config['HybridEncoder']['use_encoder_idx'] = list(use_encoder_idx)
    if 'RTDETRTransformer' in global_config:
        global_config['RTDETRTransformer']['feat_strides'] = list(feat_strides)
        global_config['RTDETRTransformer']['num_levels'] = num_levels

    print('[Ablation] use_4_scale={} -> return_idx={}, use_encoder_idx={}, '
          'feat_strides={}, num_levels={}'.format(
              use_4_scale, return_idx, use_encoder_idx, feat_strides,
              num_levels))
    return use_4_scale


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
                 exclude_post_process=False,
                 use_mkse=True,
                 use_amfem=True,
                 use_4_scale=True):
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
        self.use_mkse = use_mkse
        self.use_amfem = use_amfem
        self.use_4_scale = use_4_scale

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # Scale switch must run before creating backbone / neck / transformer
        use_4_scale = cfg.get('use_4_scale', True)
        _apply_scale_config(use_4_scale)

        backbone = create(cfg['backbone'])
        kwargs = {'input_shape': backbone.out_shape}

        # Ablation switches: use_mkse / use_amfem in DETR yaml
        use_mkse = cfg.get('use_mkse', True)
        use_amfem = cfg.get('use_amfem', True)
        # 3-scale has no C2; force disable MKSE
        if not use_4_scale and use_mkse:
            print('[Ablation] use_4_scale=False, force use_mkse=False')
            use_mkse = False

        c2_enhancer = None
        if use_mkse and cfg.get('c2_enhancer'):
            c2_enhancer = create(cfg['c2_enhancer'], **kwargs)
            kwargs = {'input_shape': c2_enhancer.out_shape}

        c5_enhancer = None
        if use_amfem and cfg.get('c5_enhancer'):
            c5_enhancer = create(cfg['c5_enhancer'], **kwargs)
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

        kwargs = {
            'hidden_dim': transformer.hidden_dim,
            'nhead': transformer.nhead,
            'input_shape': backbone.out_shape
        }
        detr_head = create(cfg['detr_head'], **kwargs)

        print('[Ablation] use_mkse={}, use_amfem={}, '
              'c2_enhancer={}, c5_enhancer={}'.format(
                  use_mkse, use_amfem,
                  type(c2_enhancer).__name__ if c2_enhancer else None,
                  type(c5_enhancer).__name__ if c5_enhancer else None))

        return {
            'backbone': backbone,
            'c2_enhancer': c2_enhancer,
            'c5_enhancer': c5_enhancer,
            'transformer': transformer,
            "detr_head": detr_head,
            "neck": neck
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)

        if self.c2_enhancer is not None:
            body_feats = self.c2_enhancer(body_feats)
        if self.c5_enhancer is not None:
            body_feats = self.c5_enhancer(body_feats)

        if self.neck is not None:
            body_feats = self.neck(body_feats)

        pad_mask = self.inputs.get('pad_mask', None)
        out_transformer = self.transformer(body_feats, pad_mask, self.inputs)

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
