#!/usr/bin/env python3
"""Measure Params(M) and FLOPs(G) for a PaddleDetection model."""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppdet.core.workspace import load_config, create
from ppdet.engine.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-o', '--override', nargs='*', default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.override:
        from ppdet.core.workspace import merge_config
        cfg = merge_config(cfg, args.override)

    cfg.print_params = False
    cfg.print_flops = False
    trainer = Trainer(cfg, mode='eval')

    params = sum(
        p.numel().item() if hasattr(p.numel(), 'item') else int(p.numel())
        for n, p in trainer.model.named_parameters()
        if all(x not in n for x in ['_mean', '_variance', 'aux_'])
    )
    print(f'Model Params : {params / 1e6:.2f} M')

    try:
        from paddleslim.analysis import dygraph_flops as calc_flops
        loader = create('EvalReader')(
            trainer.dataset, cfg.worker_num, trainer._eval_batch_sampler)
        trainer.model.eval()
        if hasattr(trainer.model, 'aux_neck'):
            trainer.model.__delattr__('aux_neck')
        if hasattr(trainer.model, 'aux_head'):
            trainer.model.__delattr__('aux_head')
        input_data = next(iter(loader))
        input_spec = [{
            'image': input_data['image'][0].unsqueeze(0),
            'im_shape': input_data['im_shape'][0].unsqueeze(0),
            'scale_factor': input_data['scale_factor'][0].unsqueeze(0),
        }]
        flops_g = calc_flops(trainer.model, input_spec) / (1000**3)
        shape = tuple(input_data['image'][0].unsqueeze(0).shape)
        print(f'Model FLOPs : {flops_g:.4f} G (image shape is {shape})')
    except ImportError:
        print('FLOPs: install paddleslim first (`pip install paddleslim`)')


if __name__ == '__main__':
    main()
