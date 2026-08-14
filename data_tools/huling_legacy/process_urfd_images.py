"""
护龄 —— URFD 图片数据集快速处理脚本

适配 D:\迅雷下载\main_data 的目录结构：
  main_data/
    train/    Fall/  +  Non_Fall/
    test/     Fall/  +  Non_Fall/
    valid/    Fall/  +  Non_Fall/

将逐帧 PNG → MediaPipe 关键点 → 特征向量 → CSV

用法：
    python process_urfd_images.py
    python process_urfd_images.py --max-frames 500   # 每类最多取500帧
    python process_urfd_images.py --skip-mediapipe   # 跳过MediaPipe（如果已有关键点缓存）
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR, STATE_NAMES, STATE_LABELS,
    CAMERA_WIDTH, CAMERA_HEIGHT, MODEL_COMPLEXITY,
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
)
from feature_extractor import FeatureExtractor, landmarks_from_mediapipe, Landmark3D


# ============================================================
# 配置
# ============================================================
DATASET_ROOT = r"D:\迅雷下载\main_data"

# 类别映射：URFD 的 Fall/Non_Fall → 护龄 6 类状态
# Non_Fall 在 URFD 中主要是行走，也有少量坐/躺
# 这里先简单映射，后续可以细化
LABEL_MAP = {
    "Fall": "fall",
    "Non_Fall": "walking",  # URFD 的日常活动主要是走动
}

# 可选：更细粒度的映射（如果你标注了具体活动）
# LABEL_MAP = {
#     "Fall": "fall",
#     "Non_Fall": "walking",
#     "Sitting": "sitting",
#     "Lying": "lying",
# }


def find_all_images(root: str, splits=("train", "test", "valid"), max_per_class=None):
    """
    遍历目录，返回 [(image_path, label, split), ...]
    """
    data = []
    for split in splits:
        split_dir = Path(root) / split
        if not split_dir.exists():
            print(f"   目录不存在: {split_dir}")
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in LABEL_MAP:
                print(f"   未知类别: {class_name}，跳过")
                continue

            label_str = LABEL_MAP[class_name]
            label_id = STATE_LABELS[label_str]

            images = sorted(class_dir.glob("*.png")) + sorted(class_dir.glob("*.jpg"))
            if max_per_class:
                images = images[:max_per_class]

            data.extend([(str(img), label_id, split) for img in images])

    return data


def process_images(image_list: List[tuple], batch_size=100):
    """
    批量处理图片：MediaPipe 提取关键点 → 特征提取。

    Returns
    -------
    (features_list, labels_list, stats_dict)
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,  # 图片模式（精度更高）
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    extractor = FeatureExtractor(use_motion=False)  # 图片互不相关，不计算运动
    features_list = []
    labels_list = []
    splits_list = []

    stats = {"total": 0, "detected": 0, "failed": 0, "per_class": {}}

    print(f"\n MediaPipe 推理 + 特征提取 ({len(image_list)} 张图片)...")

    for img_path, label_id, split in tqdm(image_list):
        stats["total"] += 1
        class_name = list(LABEL_MAP.keys())[list(LABEL_MAP.values()).index(STATE_NAMES[label_id])] \
            if STATE_NAMES[label_id] in LABEL_MAP.values() else STATE_NAMES[label_id]
        stats["per_class"].setdefault(class_name, 0)

        try:
            # 读取图片（兼容中文路径）
            frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                stats["failed"] += 1
                continue

            # 转 RGB → MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if not results.pose_landmarks:
                stats["failed"] += 1
                continue

            # 提取特征
            landmarks = landmarks_from_mediapipe(results.pose_landmarks)
            fv = extractor.extract(landmarks)

            features_list.append(fv)
            labels_list.append(label_id)
            splits_list.append(split)
            stats["detected"] += 1
            stats["per_class"][class_name] += 1

        except Exception as e:
            stats["failed"] += 1
            if stats["failed"] <= 5:  # 只打印前几个错误
                print(f"\n   处理失败: {img_path} → {e}")

    pose.close()
    return features_list, labels_list, splits_list, stats


def save_csv(features_list, labels_list, splits_list, output_path, feature_names):
    """保存为标准 CSV（train_model.py 兼容格式）"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = list(feature_names) + ["label", "label_name", "split"]
        writer.writerow(header)
        for fv, label, split in zip(features_list, labels_list, splits_list):
            row = list(fv.values) + [label, STATE_NAMES[label], split]
            writer.writerow(row)
    return len(features_list)


def main():
    parser = argparse.ArgumentParser(description="URFD 图片数据集处理")
    parser.add_argument("--data-root", type=str, default=DATASET_ROOT,
                        help="数据集根目录")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 名称")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="每类最多取多少帧（默认不限制）")
    parser.add_argument("--splits", type=str, default="train,test",
                        help="处理哪些 split（逗号分隔，默认 train,test）")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]

    print("=" * 60)
    print("  URFD 图片数据集 → 护龄特征 CSV")
    print("=" * 60)
    print(f"  数据目录: {args.data_root}")
    print(f"  处理 splits: {splits}")
    if args.max_frames:
        print(f"  每类上限: {args.max_frames} 帧")

    # 1. 扫描图片
    print(f"\n 步骤 1/3: 扫描图片...")
    image_list = find_all_images(args.data_root, splits=splits, max_per_class=args.max_frames)

    # 统计
    from collections import Counter
    split_counts = Counter(s[2] for s in image_list)
    class_counts = Counter(STATE_NAMES[s[1]] for s in image_list)
    print(f"  共找到 {len(image_list)} 张图片")
    for split_name in splits:
        print(f"    {split_name}: {split_counts.get(split_name, 0)} 张")
    print(f"  类别分布:")
    for cls_name, count in class_counts.most_common():
        print(f"    {cls_name}: {count} 张")

    if len(image_list) == 0:
        print("\n 没有找到图片！请检查目录路径")
        sys.exit(1)

    # 2. 处理图片
    print(f"\n 步骤 2/3: 处理图片...")
    t0 = time.time()
    features_list, labels_list, splits_list, stats = process_images(image_list)
    elapsed = time.time() - t0

    detection_rate = stats["detected"] / stats["total"] * 100 if stats["total"] > 0 else 0
    print(f"\n   完成!")
    print(f"  总图片: {stats['total']}")
    print(f"  检测到人体: {stats['detected']} ({detection_rate:.1f}%)")
    print(f"  未检测到: {stats['failed']}")
    print(f"  用时: {elapsed:.0f} 秒 ({stats['detected']/elapsed:.1f} 帧/秒)" if elapsed > 0 else "")

    # 3. 保存
    print(f"\n 步骤 3/3: 保存 CSV...")
    if args.output is None:
        args.output = f"urfd_features_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    if not os.path.isabs(args.output):
        args.output = os.path.join(DATA_DIR, args.output)

    extractor = FeatureExtractor(use_motion=False)
    total = save_csv(features_list, labels_list, splits_list, args.output, extractor.feature_names())

    print(f"   已保存: {args.output}")
    print(f"  特征条数: {total}")
    print(f"  特征维度: {len(features_list[0].values) if features_list else 'N/A'}")

    # 最终统计
    print(f"\n{'=' * 60}")
    print(f"  处理完成!")
    print(f"\n  下一步:")
    print(f"    python train_model.py --csv data/{os.path.basename(args.output)}")
    print(f"\n   注意: URFD 只有 Fall/Non_Fall 两类，训练出来是二分类模型。")
    print(f"    如需 6 类模型，需要补充更多数据（SisFall 或自己录制）。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
