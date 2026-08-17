# RT-DETR Full Model 复杂度与速度指标

**Config:** `configs/rtdetr/rtdetr_r50vd_6x_coco.yml`（继承 `_base_/rtdetr_r50vd.yml`）

**模块开关（全开）:**
| 开关 | 值 |
|------|-----|
| use_4_scale | True |
| use_c2_lsk | True |
| use_lsfem | True |
| use_dsf | True |

**权重:** `output/best_model/model.pdparams`  
**输入尺寸:** 640×640，BS=1  
**测试时间:** 2026-08-03  
**GPU:** NVIDIA GeForce RTX 3090  

---

## 汇总表

| 指标 | 数值 | 备注 |
|------|------|------|
| **Params (M)** | **54.04** | 可训练参数量，不含 BN running stats |
| **FLOPs (G)** | **170.66** | 输入 1×3×640×640，PaddleSlim dygraph_flops |
| **Latency (ms)** | **106.5** | 端到端 = 预处理 + 推理 + 后处理 |
| **FPS（端到端）** | **9.4** | 1000 / 106.5 ms |
| **Inference (ms)** | **73.6** | 仅网络推理 |
| **FPS（纯推理）** | **13.6** | 1000 / 73.6 ms |

---

## 与 RT-DETR-R50 官方基线对比

| 方法 | Params (M) | FLOPs (G) | FPS | 测试环境 |
|------|:----------:|:---------:|:---:|----------|
| RT-DETR-R50（官方） | 42 | 136 | 108 | T4, TensorRT FP16 |
| **Ours Full（C2+LSFEM+DSF）** | **54.04** | **170.66** | **13.6*** | 3090, Paddle Inference |

\* 当前环境未安装 TensorRT，FPS 为 Paddle GPU 纯推理；若需与官方对齐，需安装 TensorRT 后使用 `--run_mode=trt_fp16` 重测。

---

## 复现命令

```bash
# Params + FLOPs
/home/sysadmin/miniconda3/envs/paddledet39/bin/python tools/measure_model.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  -o weights=output/best_model/model.pdparams

# 导出 + FPS
/home/sysadmin/miniconda3/envs/paddledet39/bin/python tools/export_model.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  -o weights=output/best_model/model.pdparams trt=True \
  --output_dir=output_inference

/home/sysadmin/miniconda3/envs/paddledet39/bin/python deploy/python/infer.py \
  --model_dir=output_inference/rtdetr_r50vd_6x_coco \
  --image_dir=demo --device=GPU --run_benchmark=True --run_mode=paddle
```
