# RT-DETR-R18 Full Model 复杂度与速度指标

> 说明：代码库中为 **ResNet-18（R18）**，无 R15 配置；以下基于 `configs/rtdetr/rtdetr_r18vd_6x_coco.yml`。

**Config:** `configs/rtdetr/rtdetr_r18vd_6x_coco.yml`（继承 `_base_/rtdetr_r50vd.yml`）

**模块开关（全开）:**
| 开关 | 值 |
|------|-----|
| use_4_scale | True |
| use_c2_lsk | True |
| use_lsfem | True |
| use_dsf | True |

**R18 特有覆盖:**
- `ResNet.depth: 18`
- `LSFEM.out_channels: 128`（R50 为 512）
- `RTDETRTransformer.num_decoder_layers: 3`（R50 为 6）
- `HybridEncoder.expansion: 0.5`

**输入尺寸:** 640×640，BS=1  
**测试时间:** 2026-08-03  
**GPU:** NVIDIA GeForce RTX 3090  

---

## 汇总表（实测）

| 指标 | R18 Full（实测） | RT-DETR-R18 官方基线 | 差值 |
|------|:----------------:|:--------------------:|:----:|
| **Params (M)** | **24.67** | 20 | +4.67 (+23%) |
| **FLOPs (G)** | **60.73** | 60 | +0.73 (+1.2%) |
| **Inference (ms)** | **44.8** | — | 3090 Paddle |
| **FPS（纯推理）** | **22.3** | — | 3090 Paddle |
| **Latency (ms)** | **79.2** | — | 端到端 |
| **FPS（端到端）** | **12.6** | — | 3090 Paddle |
| **FPS（官方标准估算）** | **~214** | **217** | T4 TRT FP16 估算 |

---

## 与 R50 Full 对比

| 指标 | R18 Full | R50 Full |
|------|:--------:|:--------:|
| Params (M) | 24.67 | 54.04 |
| FLOPs (G) | 60.73 | 170.66 |
| Inference (ms, 3090) | 44.8 | 73.6 |
| FPS 纯推理 (3090) | 22.3 | 13.6 |
| FPS 估算 (T4 TRT FP16) | ~214 | ~86~95 |

---

## 官方标准 FPS 估算说明

FLOPs 与官方 R18 基线几乎相同（60.73G vs 60G），按 FLOPs 比例外推：

\[
\text{FPS}_{\text{R18 Full}} \approx 217 \times \frac{60}{60.73} \approx 214 \text{ FPS (T4 TRT FP16)}
\]

相对官方 R18 基线（217 FPS）仅慢约 **1~3 FPS（~1%）**，模块增量对 R18 几乎不增加计算量（LSFEM 通道数 128，解码器仅 3 层）。

---

## 复现命令

```bash
# Params + FLOPs
/home/sysadmin/miniconda3/envs/paddledet39/bin/python tools/measure_model.py \
  -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml

# FPS（需先导出，无权重也可测结构速度）
/home/sysadmin/miniconda3/envs/paddledet39/bin/python deploy/python/infer.py \
  --model_dir=output_inference/rtdetr_r18vd_6x_coco/rtdetr_r18vd_6x_coco \
  --image_dir=demo --device=GPU --run_benchmark=True --run_mode=paddle
```
