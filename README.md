# HREF-DETR: Hierarchical Representation Enhancement and Adaptive Feature Fusion for Real-Time Small-Object Detection in UAV Aerial Imagery

[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-2.6.2-blue.svg)](https://www.paddlepaddle.org.cn/)
[![Python](https://img.shields.io/badge/Python-3.9-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

Improved RT-DETR implementation based on [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) for small-object detection in UAV aerial imagery.

HREF-DETR introduces four-scale features, shallow-feature enhancement, and dynamic feature fusion into the feature extraction and hybrid encoder of RT-DETR-R50, while keeping the query decoder, Hungarian matching, and training objectives unchanged.

## Modules

| Module | Full name | Code implementation | Location | Main purpose |
| --- | --- | --- | --- | --- |
| MKSE | Selective-Kernel Enhancement Module | `MKSE` registered in `lskblock.py` (imported via `transformers/__init__.py`) | C2, stride 4 | Enhance shallow localization cues and suppress background texture interference |
| AMFEM | Adaptive Multi-scale Feature Enhancement Module | `AMFEM` in `amfem.py` | C3, stride 8 | Multi-scale selection, dual-pooling channel recalibration, spatial filtering, and FFN refinement |
| DSF | Dynamic Semantic Fusion Module | `DSF` in `dsf.py` | Bottom-up PAN fusion nodes | Adaptively balance semantic features and detail features |

**MKSE registration note.** Training uses the `MKSE` class defined and `@register`ed in `ppdet/modeling/transformers/lskblock.py`. `ppdet/modeling/transformers/__init__.py` imports it with `from .lskblock import *`. A standalone copy exists at `MKSE.py` for reference, but it is **not** the training registration path.

**Config key note.** `DETR.c5_enhancer: AMFEM` is the historical field name; AMFEM actually enhances **C3** (not C5).

## Highlights

- Extend RT-DETR input features from C3–C5 to four scales C2–C5.
- Apply lightweight MKSE enhancement to high-resolution C2 features.
- Apply adaptive multi-scale AMFEM enhancement to mid-level C3 features.
- Insert DSF into the bottom-up PAN fusion nodes.
- Keep the RT-DETR query decoder, matching strategy, and loss functions unchanged.
- Support VisDrone2019-DET and other COCO-format datasets.

## Environment

This project was developed and evaluated under:

| Component | Version |
| --- | --- |
| Python | 3.9 |
| PaddlePaddle | 2.6.2 |
| CUDA | 11.8 |
| cuDNN | 8.6.0 |

Install the matching PaddlePaddle GPU build for your GPU driver and CUDA environment.

## Installation

```bash
# 1. Create and activate the environment
conda create -n paddledet39 python=3.9 -y
conda activate paddledet39

# 2. Install PaddlePaddle GPU
# Follow the official instructions for your CUDA version:
# https://www.paddlepaddle.org.cn/install/quick

# 3. Clone and install the repository
git clone https://github.com/likaidi111/HREF-DETR.git
cd HREF-DETR
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
|       |   |-- rtdetr_r50vd.yml          # paper switches: MKSE / AMFEM / DSF / 4-scale
|       |   `-- rtdetr_reader.yml
|       |-- rtdetr_r50vd_6x_coco.yml      # R50 VisDrone entry (uses visdrone_detection.yml)
|       `-- rtdetr_r18vd_6x_coco.yml      # R18 VisDrone entry
|-- ppdet/modeling/
|   |-- architectures/
|   |   `-- detr.py                       # use_4_scale / use_mkse / use_amfem
|   `-- transformers/
|       |-- lskblock.py                   # MKSE registered here (training path)
|       |-- amfem.py                      # AMFEM 
|       |-- dsf.py                        # DSF 
|       `-- hybrid_encoder.py             # use_dsf
`-- tools/
    |-- train.py
    |-- eval.py
    `-- infer.py
```

## Dataset Preparation

### VisDrone2019-DET

Download the image-based object-detection subset from the [official VisDrone repository](https://github.com/VisDrone/VisDrone-Dataset), convert annotations to COCO JSON, and arrange files as follows:

```text
dataset/visdrone/
|-- VisDrone2019-DET-train/
|   |-- annotations/
|   |   `-- .txt
|   |-- images/
|   |   `-- .jpg
|-- VisDrone2019-DET-val/
|   |-- annotations/
|   |   `-- .txt
|   |-- images/
|   |   `-- .jpg
|-- VisDrone2019-DET-test_dev/
|   |-- annotations/
|   |   `-- .txt
|   |-- images/
|   |   `-- .jpg
|-- train_info.json
|-- val_info.json
`-- test_dev.json
```

The dataset configuration is `configs/datasets/visdrone_detection.yml`:

```yaml
metric: COCO
num_classes: 10

TrainDataset:
  !COCODataSet
    image_dir: VisDrone2019-DET-train
    anno_path: train_info.json
    dataset_dir: your local root directory of the VisDrone dataset
    data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  !COCODataSet
    image_dir: VisDrone2019-DET-val
    anno_path: val_info.json
    dataset_dir: your local root directory of the VisDrone dataset
    allow_empty: true

TestDataset:
  !ImageFolder
    image_dir: VisDrone2019-DET-test_dev
    anno_path: test_dev.json
    dataset_dir: your local root directory of the VisDrone dataset
```

Set `dataset_dir` to your local VisDrone root. JSON files must use valid COCO category IDs and image paths that match the directories above.

## Paper Configuration

The paper model uses four backbone outputs with strides 4, 8, 16, and 32. MKSE enhances C2, AMFEM enhances C3, and DSF is enabled in the bottom-up PAN path.

Default switches live in `configs/rtdetr/_base_/rtdetr_r50vd.yml`:

```yaml
DETR:
  backbone: ResNet
  neck: HybridEncoder
  transformer: RTDETRTransformer
  detr_head: DINOHead
  post_process: DETRPostProcess
  # ---- scale / module switches ----
  use_4_scale: True        # True=four-scale, False=three-scale (auto-aligns related fields)
  use_mkse: True           # True=enable MKSE, False=disable (forced off when use_4_scale=False)
  use_amfem: True          # True=enable AMFEM, False=disable
  c2_enhancer: MKSE        # registered class name in lskblock.py
  c5_enhancer: AMFEM       # historical key name; module enhances C3

MKSE:
  act: 'silu'

AMFEM:
  out_channels: 512
  num_blocks: 0
  ffn_expansion: 4
  dropout: 0.0
  act: 'silu'
  use_eca: True
  reduction: 16
  use_ffn: True

HybridEncoder:
  use_dsf: True
```

When `use_4_scale: True`, `detr.py` automatically sets:

- `ResNet.return_idx: [0, 1, 2, 3]`
- `HybridEncoder.use_encoder_idx: [3]`
- `RTDETRTransformer.feat_strides: [4, 8, 16, 32]`
- `RTDETRTransformer.num_levels: 4`

When `use_4_scale: False`, it sets `return_idx: [1, 2, 3]`, `use_encoder_idx: [2]`, `feat_strides: [8, 16, 32]`, and `num_levels: 3`.

Do not hand-write conflicting values for these fields.

**Entry configs**

| Model | Config |
| --- | --- |
| R50 (paper default) | `configs/rtdetr/rtdetr_r50vd_6x_coco.yml` |
| R18 (lightweight) | `configs/rtdetr/rtdetr_r18vd_6x_coco.yml` |

Both inherit `_base_/rtdetr_r50vd.yml` and `datasets/visdrone_detection.yml`.

## Ablation Configuration

Toggle modules with the boolean switches below. Keep registered names `c2_enhancer: MKSE` and `c5_enhancer: AMFEM`; enable or disable them via `use_mkse` / `use_amfem`.

| Experiment | Configuration change |
| --- | --- |
| Three-scale baseline | Set `DETR.use_4_scale: False` (auto-syncs `return_idx` to `[1, 2, 3]`, `use_encoder_idx` to `[2]`, `feat_strides` to `[8, 16, 32]`, `num_levels: 3`; MKSE is forced off because there is no C2) |
| Four-scale C2–C5 | Set `DETR.use_4_scale: True` (auto-syncs four-scale fields) |
| Disable MKSE | Set `DETR.use_mkse: False` |
| Enable MKSE | Set `DETR.use_mkse: True` and `DETR.c2_enhancer: MKSE` (requires `use_4_scale: True`) |
| Disable AMFEM | Set `DETR.use_amfem: False` |
| Enable AMFEM | Set `DETR.use_amfem: True` and `DETR.c5_enhancer: AMFEM` |
| Disable DSF | Set `HybridEncoder.use_dsf: False` |
| Enable DSF | Set `HybridEncoder.use_dsf: True` |

Full paper model: `use_4_scale=True`, `use_mkse=True`, `use_amfem=True`, `use_dsf=True`.

Startup logs print `[Ablation] ...` lines so you can verify the active switches.

## Training

### Single-GPU

```bash
export CUDA_VISIBLE_DEVICES=0

python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  --eval \
  --amp \
  --use_vdl=True \
  --vdl_log_dir=./work/
```

R18:

```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml \
  --eval \
  --amp
```

### Multi-GPU

```bash
python -m paddle.distributed.launch --gpus 0,1,2,3 \
  tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  --eval \
  --amp
```

### Resume

```bash
python tools/train.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  --eval \
  -r output/<checkpoint_prefix>
```

## Evaluation

```bash
python tools/eval.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
  -o weights=output/best_model.pdparams
```

## Inference

```bash
python tools/infer.py \
  -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
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
