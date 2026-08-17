"""
VisDrone 数据集子集生成脚本
"""

import json
import os
import random
import shutil

# ==================== 配置 ====================
DATASET_ROOT = r"d:\Pycharm Workroom\PaddleDetection\dataset\visdrone"

SUBSET_CONFIG = {
    "train": {
        "original_json": "train_info.json",
        "original_image_dir": "VisDrone2019-DET-train",
        "subset_json": "train_info_subset.json",
        "num_images": 650,
    },
    "val": {
        "original_json": "val_info.json",
        "original_image_dir": "VisDrone2019-DET-val",
        "subset_json": "val_info_subset.json",
        "num_images": 100,
    },
    "test": {
        "original_json": "test_dev.json",
        "original_image_dir": "VisDrone2019-DET-test_dev",
        "subset_json": "test_dev_subset.json",
        "num_images": 260,
    },
}

RANDOM_SEED = 42


def create_subset(config, dataset_root, split_name):
    print(f"\n{'='*60}")
    print(f"处理 {split_name} 数据集")
    print(f"{'='*60}")
    
    original_json_path = os.path.join(dataset_root, config["original_json"])
    subset_json_path = os.path.join(dataset_root, config["subset_json"])
    num_images = config["num_images"]
    
    print(f"读取原始标注文件: {original_json_path}")
    with open(original_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    print(f"原始图片数量: {len(coco_data['images'])}")
    print(f"原始标注数量: {len(coco_data['annotations'])}")
    
    available_images = len(coco_data['images'])
    if num_images > available_images:
        print(f"警告: 请求 {num_images} 张图片，但只有 {available_images} 张可用")
        num_images = available_images
    
    random.seed(RANDOM_SEED)
    selected_images = random.sample(coco_data['images'], num_images)
    selected_image_ids = set(img['id'] for img in selected_images)
    
    print(f"随机选择 {len(selected_images)} 张图片")
    
    selected_annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] in selected_image_ids
    ]
    
    print(f"对应标注数量: {len(selected_annotations)}")
    
    subset_coco = {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": coco_data['categories'],
    }
    
    for key in coco_data:
        if key not in subset_coco:
            subset_coco[key] = coco_data[key]
    
    print(f"保存子集标注文件: {subset_json_path}")
    with open(subset_json_path, 'w', encoding='utf-8') as f:
        json.dump(subset_coco, f, ensure_ascii=False)
    
    print(f"\n{split_name} 子集统计:")
    print(f"  - 图片数量: {len(subset_coco['images'])}")
    print(f"  - 标注数量: {len(subset_coco['annotations'])}")
    print(f"  - 类别数量: {len(subset_coco['categories'])}")


def main():
    print("=" * 60)
    print("VisDrone 数据集子集生成工具")
    print("=" * 60)
    print(f"数据集根目录: {DATASET_ROOT}")
    print(f"随机种子: {RANDOM_SEED}")
    
    for split_name, config in SUBSET_CONFIG.items():
        create_subset(config, DATASET_ROOT, split_name)
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print("\n生成的文件:")
    print(f"  - {os.path.join(DATASET_ROOT, 'train_info_subset.json')}")
    print(f"  - {os.path.join(DATASET_ROOT, 'val_info_subset.json')}")
    print(f"  - {os.path.join(DATASET_ROOT, 'test_dev_subset.json')}")


if __name__ == "__main__":
    main()