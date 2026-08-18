# HREF-DETR
code for HREF-DETR Hierarchical Representation Enhancement and Adaptive Feature Fusion for Real-Time Small-Object Detection in UAV Aerial Imagery 
# HREF-DETR: Hierarchical Representation Enhancement and Adaptive Feature Fusion for Real-Time Small-Object Detection in UAV Aerial Imagery 

[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-2.6.2-blue.svg)](https://www.paddlepaddle.org.cn/)
[![Python](https://img.shields.io/badge/Python-3.9-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

基于 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) 的 RT-DETR 改进实现，面向无人机航拍图像中的小目标检测,检测算法的原始代码在master分支下。

HREF-DETR 在 RT-DETR-R50 的特征提取与混合编码器中引入四尺度特征、浅层特征增强和动态特征融合，同时保持查询解码器、匈牙利匹配与训练目标不变。

## Modules

| Module | Full name | Code implementation | Location | Main purpose |
| --- | --- | --- | --- | --- |
| MKSE | Multi-Kernel Selective Enhancement Module | `C2LskEnhancer` in `lskblock.py` | C2, stride 4 | 增强浅层定位信息并抑制背景纹理干扰 |
| AMFEM | Adaptive Multi-scale Feature Enhancement Module | `AMFEM` in `lsfem.py` | C3, stride 8 | 多尺度选择、双池化通道重标定、空间筛选与FFN细化 |
| DSF | Dynamic Semantic Fusion Module | `DSF` in `dsf.py` | Bottom-up PAN fusion nodes | 自适应平衡语义特征与细节特征 |


## Highlights

- 将 RT-DETR 的输入特征由 C3-C5 扩展为 C2-C5 四个尺度。
- 使用 MKSE 对高分辨率 C2 特征进行轻量增强。
- 使用 AMFEM 对 C3 中层特征进行自适应多尺度增强。
- 在三个 PAN 自底向上融合节点中加入 DSF。
- 保持 RT-DETR 查询解码器、匹配策略和损失函数不变。
- 支持 VisDrone2019-DET，并可适配其他 COCO 格式数据集。

## Environment

本项目已使用以下环境进行开发与实验：

| Component | Version |
| --- | --- |
| Python | 3.9 |
| PaddlePaddle | 2.6.2 |
| CUDA | 11.8 |
| cuDNN | 8.6.0 |

请根据服务器显卡驱动与 CUDA 环境安装对应版本的 PaddlePaddle GPU。

## Installation

```bash
# 1. Create and activate the environment
conda create -n paddledet39 python=3.9 -y
conda activate paddledet39

# 2. Install PaddlePaddle GPU
# Follow the official instructions for your CUDA version:
# https://www.paddlepaddle.org.cn/install/quick

# 3. Clone and install the repository
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
pip install -r requirements.txt
pip install -e .
```

## Project Structure

```text
.
|-- configs/
|   |-- datasets/
|   |   `-- visdrone_detection.yml
|   `-- rtdetr/
|       |-- _base_/
|       |   |-- optimizer_6x.yml
|       |   |-- rtdetr_r50vd.yml
|       |   `-- rtdetr_reader.yml
|       `-- rtdetr_r50vd_visdrone_amfem_no_guide.yml
|-- ppdet/modeling/
|   |-- architectures/
|   |   `-- detr.py
|   `-- transformers/
|       |-- mkse.py
|       |-- amfem.py
|       |-- dsf.py
|       `-- hybrid_encoder.py
`-- tools/
    |-- train.py
    |-- eval.py
    `-- infer.py
```

## Dataset Preparation

### VisDrone2019-DET

Download the image-based object-detection subset from the [official VisDrone repository](https://github.com/VisDrone/VisDrone-Dataset), then arrange the converted COCO annotations as follows:

```text
dataset/visdrone/
|--  VisDrone2019-DET-train/
|   |-- annotations/
|   |   `-- .txt
|   |-- images/
|   |   `-- .jpg
|--  VisDrone2019-DET-val/
|   |-- annotations/
|   |   `-- .txt
|   |-- images/
|   |   `-- .jpg
|--  VisDrone2019-DET-test_dev/
|   |-- annotations/
|   |   `-- .txt
|   |-- images/
|   |   `-- .jpg
|-- train_info.json
|-- val_info.json
`-- test_dev.json
```

The dataset configuration is located at `configs/datasets/visdrone_detection.yml`:

```yaml
metric: COCO
num_classes: 10

TrainDataset:
  !COCODataSet
    image_dir: VisDrone2019-DET-train
    anno_path: train_info.json
    dataset_dir: Your storage location
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    image_dir: VisDrone2019-DET-val
    anno_path: val_info.json
    dataset_dir: Your storage location
    allow_empty: true

TestDataset:
  !ImageFolder
    image_dir: VisDrone2019-DET-test_dev
    anno_path: test_dev.json
    dataset_dir: Your storage location
```

> The JSON annotation files must use valid COCO category IDs and image paths that match the directory structure above.




## Paper Configuration

The paper model uses four backbone outputs with strides 4, 8, 16, and 32. MKSE enhances C2, AMFEM enhances C3, and DSF is enabled in the bottom-up PAN path. The optional guide path is disabled.

```yaml
DETR:
  backbone: ResNet
  neck: HybridEncoder
  transformer: RTDETRTransformer
  detr_head: DINOHead
  post_process: DETRPostProcess
  # ---- 尺度 / 模块开关 ----
  use_4_scale: True        # True=四尺度, False=三尺度（自动改齐相关项）
  use_mkse: True           # True=启用 MKSE, False=关闭（三尺度时自动关）
  use_amfem: True          # True=启用 AMFEM, False=关闭
  c2_enhancer: MKSE
  c5_enhancer: AMFEM

MKSE:
  act: 'silu'

AMFEM:
  out_channels: 512
  num_blocks: 0
  ffn_expansion: 4
  dropout: 0.0
  act: silu
  use_eca: true
  reduction: 16
  use_ffn: true

ResNet:
  return_idx: [0, 1, 2, 3]

HybridEncoder:
  use_dsf: true

RTDETRTransformer:
  feat_strides: [4, 8, 16, 32]
  num_levels: 4
```

The complete paper entry configuration is:

```text
configs/rtdetr/rtdetr_r50vd_visdrone_amfem_no_guide.yml
```

## Ablation Configuration

The current code does not use the undocumented switches `use_mkse`, `use_amfem`, or `use_4_scale`. Use separate configuration files for ablation experiments.

| Experiment | Configuration change |
| --- | --- |
| Three-scale baseline | Use C3-C5 backbone outputs and set transformer strides to `[8, 16, 32]` with `num_levels: 3` |
| Four-scale C2-C5 | Use `ResNet.return_idx: [0, 1, 2, 3]` and transformer strides `[4, 8, 16, 32]` |
| Disable MKSE | Remove `DETR.c2_enhancer` or set it to `null` in a dedicated config |
| Enable MKSE | Set `DETR.c2_enhancer: C2LskEnhancer` |
| Disable AMFEM | Remove `DETR.c5_enhancer` or set it to `null` in a dedicated config |
| Enable AMFEM | Set `DETR.c5_enhancer: AMFEM` |
| Disable DSF | Set `HybridEncoder.use_dsf: false` |
| Enable DSF | Set `HybridEncoder.use_dsf: true` |

When switching between three-scale and four-scale models, keep the backbone output shapes, hybrid encoder input levels, transformer `feat_strides`, and `num_levels` mutually consistent.


### Train

```bash
export CUDA_VISIBLE_DEVICES=0

python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_visdrone_amfem_no_guide.yml \
  --eval \
  --amp \
  --use_vdl=True \
  --vdl_log_dir=./work/
```

### Multi-GPU Training

```bash
python -m paddle.distributed.launch --gpus 0,1,2,3 \
  tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_visdrone_amfem_no_guide.yml \
  --eval \
  --amp
```

### Resume Training

```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_visdrone_amfem_no_guide.yml \
  --eval \
  -r output/<checkpoint_prefix>
```

### Eval

```bash
python tools/eval.py \
  -c configs/rtdetr/rtdetr_r50vd_visdrone_amfem_no_guide.yml \
  -o weights=output/best_model.pdparams
```

### Inference

```bash
python tools/infer.py \
  -c configs/rtdetr/rtdetr_r50vd_visdrone_amfem_no_guide.yml \
  -o weights=output/best_model.pdparams \
  --infer_img=path/to/image.jpg \
  --draw_threshold=0.40 \
  --save_results=True
```

Outputs are saved under `output/` by default.


### Checkpoint loading test

```bash
python -c "import paddle; paddle.load('output/best_model.pdparams'); print('OK')"
```

## Acknowledgements

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)

## License

This project follows the Apache License 2.0 used by PaddleDetection. Retain the original PaddleDetection copyright and license notices when redistributing modified source files.
