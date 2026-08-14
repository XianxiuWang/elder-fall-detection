"""
护龄 — 数据集预处理工具
========================
将公开数据集视频/图片 → 整理为标准训练格式

工作流：
    1. 加载 YOLO11n → 检测视频帧中的人体
    2. 裁剪人体区域，统一缩放到 224×224
    3. 按标签分类保存到 train/val/test 目录
    4. 生成数据集索引文件 dataset.json

用法：
    # 处理 URFD 数据集
    python prepare_data.py --dataset urfd

    # 处理自定义采集的数据
    python prepare_data.py --dataset custom --input_dir data/

    # 只生成划分索引（已有图片按文件夹组织好了）
    python prepare_data.py --index_only --input_dir datasets/prepared/

最终目录结构（ImageFolder 格式）：
    datasets/prepared/
    ├── train/
    │   ├── walking/      ← 行走类图片
    │   ├── sitting/      ← 坐着类图片
    │   ├── lying/        ← 躺卧类图片
    │   ├── fall/         ← 跌倒类图片
    │   └── empty/        ← 无人场景图片
    ├── val/
    │   └── (同上结构)
    └── test/
        └── (同上结构)
"""

import os
import sys
import json
import argparse
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import config


# ============================================================
# 数据集标注映射
# ============================================================

# URFD 的视频对应的标签
# URFD 跌倒视频 → 标签5(fall)，ADL视频 → 需帧级别判断
URFD_FALL_LABEL = 5      # 跌倒
URFD_ADL_LABELS = {
    # 日常活动需要帧级别分析，先用粗略规则
    "walking": 0,
    "sitting": 1,
    "lying": 2,
    "bending": 4,   # 弯腰 → 异常姿态
    "standing": 1,  # 站着不动可以归为坐着/休息类（静止状态）
}


def detect_person_crop(model, frame: np.ndarray, conf: float = None,
                       target_size: tuple = None) -> Optional[np.ndarray]:
    """
    使用 YOLO 检测人体并裁剪

    Args:
        model: YOLO 模型实例
        frame: 输入图像 (H, W, 3)
        conf: 置信度阈值

    Returns:
        裁剪后的图像 (target_size, target_size, 3) 或 None（未检测到人）
    """
    if conf is None:
        conf = config.DETECTION_CONF
    if target_size is None:
        target_size = config.CLASSIFIER_INPUT_SIZE

    results = model(frame, conf=conf, verbose=False, classes=[config.PERSON_CLASS_ID])

    if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    # 取置信度最高的人体框
    boxes = results[0].boxes
    confs = boxes.conf.cpu().numpy()
    best_idx = np.argmax(confs)

    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    # 边界检查
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)

    return crop


def process_urfd_dataset(dataset_path: Path, output_dir: Path, detection_model):
    """
    处理 UR Fall Detection Dataset

    URFD 结构:
        urfd/
        ├── urfall-cam0-falls/    ← cam0 跌倒视频
        ├── urfall-cam0-adls/     ← cam0 日常活动视频
        ├── urfall-cam1-falls/    ← cam1 跌倒视频
        └── urfall-cam1-adls/     ← cam1 日常活动视频

    每个视频文件夹包含:
        - video.avi
        - 帧图片 (frame-xxxx.png)
        - 标签文件 (falls.csv / adls.csv)
    """
    print(f"\n{'='*60}")
    print("  处理 UR Fall Detection Dataset")
    print(f"{'='*60}")

    # 收集所有视频目录
    video_dirs = []
    for pattern in ["*falls*", "*adls*"]:
        for d in dataset_path.rglob(pattern):
            if d.is_dir():
                video_dirs.append(d)

    if not video_dirs:
        print(f"❌ 未找到 URFD 视频目录！请确保数据集已下载到 {dataset_path}")
        print("   可以运行: python download_datasets.py --dataset urfd")
        return

    print(f"找到 {len(video_dirs)} 个视频目录")

    all_samples = []  # [(frame_path, label), ...]

    for vdir in tqdm(video_dirs, desc="处理URFD视频"):
        # 找到视频文件
        video_files = list(vdir.glob("*.avi")) + list(vdir.glob("*.mp4"))
        img_files = sorted(list(vdir.glob("*.png"))) + sorted(list(vdir.glob("*.jpg")))

        # 确定标签
        label_name = vdir.name.lower()
        if "fall" in label_name:
            label = 5  # 跌倒
        else:
            # ADL 视频 — 需要帧级别分析，这里简化处理
            # 取视频名做粗略判断
            label = None

        # 如果视频目录下有图片，直接处理图片
        if img_files and len(img_files) > 10:
            save_dir = output_dir / "raw" / vdir.name
            save_dir.mkdir(parents=True, exist_ok=True)

            for img_file in tqdm(img_files[::3], desc=f"  {vdir.name}", leave=False):  # 每3帧取1帧
                frame = cv2.imread(str(img_file))
                if frame is None:
                    continue

                crop = detect_person_crop(detection_model, frame)
                if crop is not None:
                    out_path = save_dir / f"{img_file.stem}_person.jpg"
                    cv2.imwrite(str(out_path), crop)
                    all_samples.append((out_path, label if label is not None else 1))  # 默认坐姿

        else:
            # 读视频
            for vf in video_files:
                cap = cv2.VideoCapture(str(vf))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                save_dir = output_dir / "raw" / f"{vf.stem}"
                save_dir.mkdir(parents=True, exist_ok=True)

                frame_idx = 0
                sample_every = max(1, int(fps / 5))  # 每秒取5帧

                pbar = tqdm(total=total_frames, desc=f"  {vf.name}", leave=False)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % sample_every == 0:
                        crop = detect_person_crop(detection_model, frame)
                        if crop is not None:
                            out_path = save_dir / f"frame_{frame_idx:06d}.jpg"
                            cv2.imwrite(str(out_path), crop)
                            all_samples.append((out_path, label if label is not None else 1))

                    frame_idx += 1
                    pbar.update(1)

                pbar.close()
                cap.release()

    print(f"\n📊 总计提取 {len(all_samples)} 个有效人体样本")

    # 标签分布
    label_counts = defaultdict(int)
    for _, label in all_samples:
        label_counts[label] += 1
    print("\n标签分布:")
    for lbl, count in sorted(label_counts.items()):
        print(f"  [{lbl}] {config.STATE_NAMES.get(lbl, f'未知')}: {count} 张")

    return all_samples


def process_custom_dataset(input_dir: Path, output_dir: Path, detection_model):
    """
    处理自定义数据集（你自己录的视频/图片）
    期望输入结构：
        input_dir/
        ├── walking/    ← 行走视频/图片
        ├── sitting/    ← 坐着视频/图片
        ├── lying/      ← 躺卧视频/图片
        ├── fall/       ← 跌倒视频/图片
        └── empty/      ← 无人场景
    """
    print(f"\n{'='*60}")
    print("  处理自定义数据集")
    print(f"{'='*60}")

    label_map = {
        "walking": 0, "walk": 0,
        "sitting": 1, "sit": 1,
        "lying": 2, "lie": 2, "sleep": 2,
        "sedentary": 3,
        "abnormal": 4, "bend": 4, "bending": 4,
        "fall": 5, "fallen": 5, "falls": 5,
        "empty": 6, "none": 6, "nobody": 6,
    }

    all_samples = []

    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # 匹配标签
        label = None
        for key, val in label_map.items():
            if key in subdir.name.lower():
                label = val
                break

        if label is None:
            print(f"  ⚠️ 跳过未知类别目录: {subdir.name}")
            continue

        print(f"  处理 [{label}] {config.STATE_NAMES[label]} ← {subdir.name}")
        save_dir = output_dir / "raw" / subdir.name
        save_dir.mkdir(parents=True, exist_ok=True)

        # 如果是无人场景（empty），直接复制图片，不需要人体检测
        if label == 6:
            for img_file in list(subdir.glob("*.jpg")) + list(subdir.glob("*.png")):
                frame = cv2.imread(str(img_file))
                if frame is not None:
                    frame = cv2.resize(frame, config.CLASSIFIER_INPUT_SIZE)
                    out_path = save_dir / img_file.name
                    cv2.imwrite(str(out_path), frame)
                    all_samples.append((out_path, label))
            continue

        # 处理图片
        img_files = list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
        for img_file in img_files:
            frame = cv2.imread(str(img_file))
            if frame is None:
                continue
            crop = detect_person_crop(detection_model, frame)
            if crop is not None:
                out_path = save_dir / img_file.name
                cv2.imwrite(str(out_path), crop)
                all_samples.append((out_path, label))

        # 处理视频
        video_files = list(subdir.glob("*.mp4")) + list(subdir.glob("*.avi"))
        for vf in video_files:
            cap = cv2.VideoCapture(str(vf))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            sample_every = max(1, int(fps / 3))

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_every == 0:
                    crop = detect_person_crop(detection_model, frame)
                    if crop is not None:
                        out_path = save_dir / f"{vf.stem}_{frame_idx:06d}.jpg"
                        cv2.imwrite(str(out_path), crop)
                        all_samples.append((out_path, label))
                frame_idx += 1
            cap.release()

    print(f"\n📊 总计提取 {len(all_samples)} 个样本")
    return all_samples


def split_and_organize(all_samples: List[Tuple[Path, int]], output_dir: Path):
    """
    将样本按 train/val/test 划分，组织为 ImageFolder 格式
    """
    print(f"\n{'='*60}")
    print("  数据划分 (train:val:test = 7:1.5:1.5)")
    print(f"{'='*60}")

    random.seed(config.RANDOM_STATE)
    random.shuffle(all_samples)

    # 按标签分组
    by_label = defaultdict(list)
    for path, label in all_samples:
        by_label[label].append((path, label))

    # 划分并复制
    for split_name, split_ratio in [("train", 0.70), ("val", 0.15), ("test", 0.15)]:
        for label, items in by_label.items():
            n = int(len(items) * split_ratio)
            start = 0 if split_name == "train" else \
                    int(len(items) * 0.70) if split_name == "val" else \
                    int(len(items) * 0.85)

            if split_name == "train":
                selected = items[:int(len(items) * split_ratio)]
            elif split_name == "val":
                selected = items[int(len(items) * 0.70):int(len(items) * 0.85)]
            else:
                selected = items[int(len(items) * 0.85):]

            state_en = config.STATE_NAMES_EN[label]

            # 创建目标目录
            split_dir = output_dir / split_name / state_en
            split_dir.mkdir(parents=True, exist_ok=True)

            # 复制文件
            for src_path, _ in selected:
                # 去重文件名
                dst_path = split_dir / f"label{label}_{Path(src_path).name}"
                try:
                    shutil.copy2(src_path, dst_path)
                except shutil.SameFileError:
                    pass

            print(f"  [{split_name:5s}] {state_en:12s}: {len(selected)} 张")

    # 生成数据集统计
    stats = {}
    for split_name in ["train", "val", "test"]:
        stats[split_name] = {}
        for label, state_en in config.STATE_NAMES_EN.items():
            split_dir = output_dir / split_name / state_en
            count = len(list(split_dir.glob("*"))) if split_dir.exists() else 0
            stats[split_name][state_en] = count

    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 数据集准备完成！")
    print(f"  输出目录: {output_dir}")
    print(f"  索引文件: {stats_file}")
    print(f"\n总样本分布:")
    for split in ["train", "val", "test"]:
        total = sum(stats[split].values())
        print(f"  {split}: {total} 张")


def main():
    parser = argparse.ArgumentParser(description="护龄 数据集预处理")
    parser.add_argument("--dataset", type=str, default="urfd",
                        choices=["urfd", "le2i", "custom"],
                        help="数据集来源 (默认: urfd)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="输入目录（custom数据集时必填）")
    parser.add_argument("--output_dir", type=str,
                        default=str(config.DATASET_DIR / "prepared"),
                        help="输出目录")
    parser.add_argument("--index_only", action="store_true",
                        help="仅划分已有目录（不重新提取人体）")
    parser.add_argument("--detection_model", type=str,
                        default=config.DETECTION_MODEL,
                        help="人体检测模型")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # 加载 YOLO 检测模型
    if not args.index_only:
        print("加载人体检测模型...")
        from ultralytics import YOLO
        try:
            det_model = YOLO(args.detection_model)
        except Exception as e:
            print(f"❌ 加载检测模型失败: {e}")
            print("尝试安装: pip install ultralytics")
            return
        print(f"✅ 已加载: {args.detection_model}")

    # 处理数据集
    if args.index_only:
        all_samples = []
        raw_dir = output_dir / "raw"
        if not raw_dir.exists():
            print(f"❌ 未找到 {raw_dir}，请先运行预处理")
            return
        for img_file in raw_dir.rglob("*.jpg"):
            # 从文件路径推断标签
            label = None
            for lbl_id in range(7):
                if f"label{lbl_id}" in str(img_file) or config.STATE_NAMES_EN[lbl_id] in str(img_file):
                    label = lbl_id
                    break
            if label is not None:
                all_samples.append((img_file, label))

    elif args.dataset == "urfd":
        urfd_path = config.DATASET_DIR / "urfd"
        all_samples = process_urfd_dataset(urfd_path, output_dir, det_model)

    elif args.dataset == "custom":
        if not args.input_dir:
            print("❌ 使用 --dataset custom 时必须指定 --input_dir")
            return
        all_samples = process_custom_dataset(Path(args.input_dir), output_dir, det_model)
    else:
        print(f"❌ 不支持的数据集: {args.dataset}")
        return

    if not all_samples:
        print("❌ 没有提取到有效样本！")
        print("可能原因:")
        print("  1. 数据集未下载 → 运行 python download_datasets.py")
        print("  2. YOLO 检测不到人体 → 调整 DETECTION_CONF 阈值")
        print("  3. 视频格式不支持 → 检查视频是否可正常播放")
        return

    # 划分数据
    split_and_organize(all_samples, output_dir)

    print(f"\n下一步: python train_classifier.py")


if __name__ == "__main__":
    main()
