"""
护龄 —— 开源数据集适配器

将常见的跌倒检测开源数据集转换为护龄模型的训练格式：
  - UR Fall Detection Dataset (URFD)
  - Le2i Fall Detection Dataset
  - SisFall Dataset
  - 自定义视频 + 标签格式

输出: 标准 CSV 文件，可直接被 train_model.py 消费。

用法:
    # UR Fall Dataset
    python dataset_loader.py --dataset urfd \\
        --data-dir D:/datasets/URFD --output urfd_features.csv

    # Le2i Dataset
    python dataset_loader.py --dataset le2i \\
        --data-dir D:/datasets/Le2i --output le2i_features.csv

    # SisFall Dataset
    python dataset_loader.py --dataset sisfall \\
        --data-dir D:/datasets/SisFall --output sisfall_features.csv

    # 通用格式（视频文件夹按类别组织）
    python dataset_loader.py --dataset generic \\
        --data-dir D:/datasets/my_data \\
        --class-map walking:0,sitting:1,fall:5 --output my_features.csv

    # 从已有关键点 JSON 加载（跳过 MediaPipe 推理）
    python dataset_loader.py --dataset pre_extracted \\
        --json-dir D:/datasets/keypoints --output features.csv

    # 仅提取关键点保存（不转换特征）
    python dataset_loader.py --dataset urfd \\
        --data-dir D:/datasets/URFD --save-keypoints keypoints.json
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from config import (
    DATA_DIR, STATE_NAMES, STATE_LABELS,
    CAMERA_WIDTH, CAMERA_HEIGHT, MODEL_COMPLEXITY,
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
)
from feature_extractor import FeatureExtractor, landmarks_from_mediapipe, Landmark3D


# ============================================================
# 工具函数
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def list_video_files(root_dir, extensions=(".mp4", ".avi", ".mov", ".mkv", ".webm")):
    """递归查找所有视频文件"""
    videos = []
    for ext in extensions:
        videos.extend(Path(root_dir).rglob(f"*{ext}"))
        videos.extend(Path(root_dir).rglob(f"*{ext.upper()}"))
    return sorted(set(videos))


def extract_frames(video_path, stride=3, max_frames=None, resize=(640, 480)):
    """
    从视频中提取帧。

    Parameters
    ----------
    video_path : 视频路径
    stride : 每隔多少帧取一帧（减少冗余，3=每3帧取1帧）
    max_frames : 最多提取帧数
    resize : 缩放尺寸 (w, h)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames = []
    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_idx % stride == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            frames.append(frame)
        frame_idx += 1
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return frames


def extract_landmarks_from_frames(frames, pose_model, verbose=False):
    """
    批量从帧中提取 MediaPipe 关键点。

    Returns
    -------
    List[List[Landmark3D]] 或空列表（无人检测到的帧返回 None 占位）
    """
    all_landmarks = []
    desc = "MediaPipe 推理" if verbose else None
    iterator = tqdm(frames, desc=desc, disable=not verbose) if verbose else frames

    for frame in iterator:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = pose_model.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if results.pose_landmarks:
            lm = landmarks_from_mediapipe(results.pose_landmarks)
            all_landmarks.append(lm)
        else:
            all_landmarks.append(None)

    return all_landmarks


def features_to_csv(features_list, labels, output_path, feature_names):
    """将特征 + 标签写入 CSV"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = list(feature_names) + ["label", "label_name"]
        writer.writerow(header)
        for fv, label in zip(features_list, labels):
            row = list(fv.values) + [label, STATE_NAMES[label]]
            writer.writerow(row)
    return len(features_list)


# ============================================================
# 数据集解析器
# ============================================================
class DatasetParser:
    """基类"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    def parse(self) -> List[Tuple[List, int]]:
        """
        解析数据集，返回 [(frames, label), ...]
        """
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__


class URFDParser(DatasetParser):
    """
    UR Fall Detection Dataset 解析器。

    预期目录结构:
      URFD/
        ├── urfall-cam0-falls/        # 跌倒视频
        │   ├── fall-01-cam0-d.mp4
        │   └── ...
        ├── urfall-cam0-adls/         # 日常活动视频
        │   ├── adl-01-cam0-d.mp4
        │   └── ...
        └── urfall-cam0-trips/        # (可选) 绊倒视频 → 归为 fall
    """

    MAP_SUBDIR = {
        "falls": "fall",
        "adls": "walking",   # URFD 的 ADL 主要是走动
        "trips": "fall",
    }

    def parse(self):
        data = []
        for subdir_name, state_name in self.MAP_SUBDIR.items():
            subdir = self.data_dir / subdir_name
            if not subdir.exists():
                # URFD 可能有不同命名
                for pattern in [f"*{subdir_name}*", f"*-{subdir_name}*"]:
                    matches = list(self.data_dir.glob(pattern))
                    if matches:
                        subdir = matches[0]
                        break
                if not subdir.exists():
                    print(f"  ⚠ 目录不存在，跳过: {subdir_name}")
                    continue

            videos = list_video_files(subdir)
            print(f"  📁 {subdir_name} ({state_name}): {len(videos)} 个视频")

            for video_path in videos:
                frames = extract_frames(video_path, stride=3, max_frames=200)
                if frames:
                    data.append((frames, STATE_LABELS[state_name]))

        print(f"  ✅ 共加载 {len(data)} 个视频片段")
        return data


class Le2iParser(DatasetParser):
    """
    Le2i Fall Detection Dataset 解析器。

    预期目录结构:
      Le2i/
        ├── Fall/
        │   ├── Fall1*.avi
        │   └── ...
        └── NotFall/
            ├── NotFall1*.avi
            └── ...

    或:
      Le2i/
        ├── Home_01/
        │   ├── Video/
        │   │   ├── fall*.avi
        │   │   └── ...
        │   └── Annotation_files/
    """

    MAP_DIRNAME = {
        "fall": "fall",
        "notfall": "walking",
        "not_fall": "walking",
        "adl": "walking",
        "walking": "walking",
    }

    def parse(self):
        data = []

        # 先尝试直接子目录匹配
        for subdir in self.data_dir.iterdir():
            if not subdir.is_dir():
                continue
            dir_lower = subdir.name.lower()

            # 匹配类别
            label = None
            for keyword, state in self.MAP_DIRNAME.items():
                if keyword in dir_lower:
                    label = STATE_LABELS[state]
                    break

            if label is None:
                # 尝试深层查找
                video_dir = subdir / "Video"
                if video_dir.exists():
                    subdir = video_dir
                    for keyword, state in self.MAP_DIRNAME.items():
                        if keyword in dir_lower:
                            label = STATE_LABELS[state]
                            break

            if label is None:
                print(f"  ⚠ 无法识别类别，跳过: {subdir.name}")
                continue

            videos = list_video_files(subdir)
            print(f"  📁 {subdir.name} → {STATE_NAMES[label]}: {len(videos)} 个视频")

            for video_path in videos:
                frames = extract_frames(video_path, stride=3, max_frames=200)
                if frames:
                    data.append((frames, label))

        print(f"  ✅ 共加载 {len(data)} 个视频片段")
        return data


class SisFallParser(DatasetParser):
    """
    SisFall Dataset 解析器。

    预期目录结构:
      SisFall/
        ├── SA01/   # 日常活动
        ├── SA02/
        ├── ...
        ├── SE01/   # 跌倒事件
        ├── SE02/
        └── ...

    SA = 日常活动 → walking / sitting
    SE = 跌倒事件 → fall

    可根据 SisFall 的标注文档进一步细分。
    """

    # SisFall 的活动分类（简化映射）
    # SA01-SA06: 慢走/快走/慢跑等 → walking
    # SA07-SA12: 坐/蹲/躺等 → sitting/lying
    # SA13-SA20: 其他活动 → walking
    # SE01-SE15: 各类跌倒 → fall
    ACTIVITY_MAP = {}

    @classmethod
    def _init_map(cls):
        if cls.ACTIVITY_MAP:
            return
        # SA01-SA06 → walking (走/跑)
        for i in range(1, 7):
            cls.ACTIVITY_MAP[f"SA{i:02d}"] = "walking"
        # SA07-SA08 → sitting
        for i in range(7, 9):
            cls.ACTIVITY_MAP[f"SA{i:02d}"] = "sitting"
        # SA09-SA10 → lying
        for i in range(9, 11):
            cls.ACTIVITY_MAP[f"SA{i:02d}"] = "lying"
        # SA11-SA20 → walking (默认)
        for i in range(11, 21):
            cls.ACTIVITY_MAP[f"SA{i:02d}"] = "walking"
        # SA21-SA24 → abnormal (蹲下/弯腰)
        for i in range(21, 25):
            cls.ACTIVITY_MAP[f"SA{i:02d}"] = "abnormal"
        # SE01-SE15 → fall
        for i in range(1, 16):
            cls.ACTIVITY_MAP[f"SE{i:02d}"] = "fall"

    def parse(self):
        self._init_map()
        data = []

        for subdir in sorted(self.data_dir.iterdir()):
            if not subdir.is_dir():
                continue
            activity_code = subdir.name.upper()[:4]
            state_name = self.ACTIVITY_MAP.get(activity_code)

            if state_name is None:
                # 回退：SE → fall, SA → walking
                if activity_code.startswith("SE"):
                    state_name = "fall"
                elif activity_code.startswith("SA"):
                    state_name = "walking"
                else:
                    print(f"  ⚠ 未知活动码: {activity_code}，跳过")
                    continue

            label = STATE_LABELS[state_name]
            videos = list_video_files(subdir)
            print(f"  📁 {subdir.name} → {state_name}: {len(videos)} 个视频")

            for video_path in videos:
                frames = extract_frames(video_path, stride=5, max_frames=150)
                if frames:
                    data.append((frames, label))

        print(f"  ✅ 共加载 {len(data)} 个视频片段")
        return data


class GenericParser(DatasetParser):
    """
    通用数据集解析器——视频按类别文件夹组织。

    预期目录结构:
      data_dir/
        ├── walking/    # 所有行走视频
        │   ├── vid1.mp4
        │   └── ...
        ├── sitting/
        ├── lying/
        ├── fall/
        └── ...

    或提供一个 class_map 手动映射文件夹名→状态。
    """

    def __init__(self, data_dir: str, class_map: Optional[Dict[str, str]] = None):
        super().__init__(data_dir)
        self.class_map = class_map or {}

    def parse(self):
        data = []

        for subdir in sorted(self.data_dir.iterdir()):
            if not subdir.is_dir():
                continue
            folder_name = subdir.name.lower()

            # 先查自定义映射，再查默认
            if folder_name in self.class_map:
                state_name = self.class_map[folder_name]
            elif folder_name in STATE_LABELS:
                state_name = folder_name
            else:
                print(f"  ⚠ 未知文件夹类别: {subdir.name}，跳过（可用 --class-map 指定映射）")
                continue

            label = STATE_LABELS[state_name]
            videos = list_video_files(subdir)
            print(f"  📁 {subdir.name} → {state_name}: {len(videos)} 个视频")

            for video_path in videos:
                frames = extract_frames(video_path, stride=3, max_frames=200)
                if frames:
                    data.append((frames, label))

        print(f"  ✅ 共加载 {len(data)} 个视频片段")
        return data


# ============================================================
# 主处理流程
# ============================================================
def process_dataset(parser: DatasetParser,
                    output_path: str,
                    save_keypoints_path: Optional[str] = None,
                    feature_stride: int = 1,
                    max_frames_per_video: int = 200) -> str:
    """
    完整处理流程：
      1. 解析数据集 → 视频帧
      2. MediaPipe 提取关键点
      3. 特征提取
      4. 保存 CSV / 关键点 JSON

    Returns
    -------
    output_path
    """
    print(f"\n{'=' * 60}")
    print(f"  数据集适配器: {parser.name}")
    print(f"  数据目录: {parser.data_dir}")
    print(f"{'=' * 60}")

    # 1. 解析数据集
    print(f"\n🔍 步骤 1/4: 解析数据集...")
    video_data = parser.parse()

    if not video_data:
        raise ValueError("没有找到任何视频数据！请检查目录结构和 --dataset 参数")

    total_frames_raw = sum(len(frames) for frames, _ in video_data)
    print(f"  总计: {len(video_data)} 个视频片段, ~{total_frames_raw} 帧")

    # 2. MediaPipe 推理
    print(f"\n🔍 步骤 2/4: MediaPipe 姿态提取...")
    print(f"  (这可能需要几分钟，取决于视频数量)")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    all_landmarks = []  # [([Landmark3D, ...], label), ...]
    total_detected = 0
    total_frames = 0

    for frames, label in tqdm(video_data, desc="处理视频"):
        # 跳帧采样
        sampled_frames = frames[::feature_stride]
        if max_frames_per_video:
            sampled_frames = sampled_frames[:max_frames_per_video]

        landmarks = extract_landmarks_from_frames(sampled_frames, pose, verbose=False)

        for lm in landmarks:
            total_frames += 1
            if lm is not None:
                all_landmarks.append((lm, label))
                total_detected += 1

    pose.close()

    detection_rate = total_detected / total_frames * 100 if total_frames > 0 else 0
    print(f"\n  ✅ 有效帧: {total_detected}/{total_frames} ({detection_rate:.1f}%)")

    if total_detected == 0:
        raise ValueError(
            "没有检测到任何人体关键点！\n"
            "可能原因: 视频中没有人、视频格式不兼容、MediaPipe 配置问题"
        )

    # 3. 特征提取
    print(f"\n🔍 步骤 3/4: 特征提取...")
    extractor = FeatureExtractor(use_motion=False)

    features_list = []
    labels_list = []
    for lm, label in tqdm(all_landmarks, desc="提取特征"):
        try:
            fv = extractor.extract(lm)
            features_list.append(fv)
            labels_list.append(label)
        except Exception as e:
            print(f"\n  ⚠ 特征提取失败 (label={label}): {e}")

    # 4. 保存
    print(f"\n🔍 步骤 4/4: 保存结果...")

    # 保存 CSV
    total_saved = features_to_csv(
        features_list, labels_list, output_path,
        extractor.feature_names()
    )
    print(f"  ✅ 特征 CSV: {output_path} ({total_saved} 条)")

    # 保存关键点（可选）
    if save_keypoints_path:
        keypoints_data = []
        for lm, label in all_landmarks:
            kp = {
                "label": int(label),
                "label_name": STATE_NAMES[label],
                "landmarks": [
                    {"x": p.x, "y": p.y, "z": p.z, "visibility": p.visibility}
                    for p in lm
                ]
            }
            keypoints_data.append(kp)

        with open(save_keypoints_path, 'w', encoding='utf-8') as f:
            json.dump(keypoints_data, f, ensure_ascii=False)
        print(f"  ✅ 关键点 JSON: {save_keypoints_path}")

    # 类分布
    print(f"\n📊 各类别分布:")
    for i, name in enumerate(STATE_NAMES):
        count = sum(1 for l in labels_list if l == i)
        if count > 0:
            print(f"    {name}: {count} 条 ({count/len(labels_list)*100:.1f}%)")

    return output_path


def parse_class_map(map_str: str) -> Dict[str, str]:
    """解析 --class-map 参数: 'folder1:state1,folder2:state2'"""
    result = {}
    if not map_str:
        return result
    for item in map_str.split(","):
        parts = item.strip().split(":")
        if len(parts) == 2:
            folder, state = parts[0].strip(), parts[1].strip()
            if state not in STATE_LABELS:
                print(f"⚠ 未知状态: {state}, 可用: {list(STATE_LABELS.keys())}")
                continue
            result[folder.lower()] = state
    return result


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="护龄 - 开源数据集适配器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python dataset_loader.py --dataset urfd --data-dir D:/datasets/URFD
  python dataset_loader.py --dataset le2i --data-dir D:/datasets/Le2i
  python dataset_loader.py --dataset sisfall --data-dir D:/datasets/SisFall
  python dataset_loader.py --dataset generic --data-dir D:/my_videos \\
      --class-map "walking:walking,fall:fall,sit:sitting"
        """
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["urfd", "le2i", "sisfall", "generic"],
                        help="数据集类型")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="数据集根目录（单目录模式）")
    parser.add_argument("--data-dirs", type=str, nargs="+", default=None,
                        help="多个数据集根目录（多目录模式，如: --data-dirs 'E:/main_data' 'E:/Three Classes'）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 路径（默认自动命名）")
    parser.add_argument("--save-keypoints", type=str, default=None,
                        help="额外保存关键点 JSON 路径")
    parser.add_argument("--class-map", type=str, default=None,
                        help="自定义类别映射（仅 generic 模式）")
    parser.add_argument("--stride", type=int, default=1,
                        help="跳帧采样间隔（默认 1=每帧都取；2=隔1帧取1帧）")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="每个视频最多取多少帧")
    args = parser.parse_args()

    # 验证：data-dir 和 data-dirs 至少提供一个
    if not args.data_dir and not args.data_dirs:
        parser.error("必须指定 --data-dir 或 --data-dirs 参数")

    # 收集所有数据目录
    all_data_dirs = []
    if args.data_dir:
        all_data_dirs.append(args.data_dir)
    if args.data_dirs:
        all_data_dirs.extend(args.data_dirs)

    # 输出路径
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(DATA_DIR, f"{args.dataset}_features_{timestamp}.csv")
    elif not os.path.isabs(args.output):
        args.output = os.path.join(DATA_DIR, args.output)

    # 如果多个数据目录，逐个处理并合并
    if len(all_data_dirs) > 1:
        print(f"\n 多目录模式: {len(all_data_dirs)} 个数据源")
        for i, d in enumerate(all_data_dirs):
            print(f"   [{i+1}] {d}")

        # 处理每个目录，合并结果
        import pandas as pd
        all_csv_parts = []
        for idx, data_dir in enumerate(all_data_dirs):
            part_output = args.output.replace('.csv', f'_part{idx}.csv')
            print(f"\n{'#'*60}")
            print(f"# 处理数据源 {idx+1}/{len(all_data_dirs)}: {data_dir}")
            print(f"{'#'*60}")

            _process_single_dataset(
                args.dataset, data_dir, part_output,
                args.save_keypoints, args.stride, args.max_frames,
                args.class_map
            )
            if os.path.exists(part_output):
                all_csv_parts.append(part_output)

        # 合并所有部分
        if all_csv_parts:
            print(f"\n{'='*60}")
            print(f"  合并 {len(all_csv_parts)} 个数据源...")
            dfs = []
            for part_path in all_csv_parts:
                df = pd.read_csv(part_path)
                dfs.append(df)
                os.remove(part_path)  # 删除中间文件
            merged = pd.concat(dfs, ignore_index=True)
            merged.to_csv(args.output, index=False, encoding='utf-8')
            print(f"  ✅ 合并完成: {args.output} ({len(merged)} 条)")
            print(f"{'='*60}")
        return

    # 单目录模式
    data_dir = all_data_dirs[0]
    if not os.path.isdir(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return

    # 选择解析器
    data_parser = _create_parser(args.dataset, data_dir, args.class_map)

    # 处理
    t0 = time.time()
    try:
        output_path = process_dataset(
            data_parser,
            args.output,
            save_keypoints_path=args.save_keypoints,
            feature_stride=args.stride,
            max_frames_per_video=args.max_frames,
        )
        elapsed = time.time() - t0

        print(f"\n{'=' * 60}")
        print(f"  ✅ 处理完成！")
        print(f"  总用时: {elapsed:.1f} 秒")
        print(f"  输出文件: {output_path}")
        print(f"\n  下一步:")
        print(f"    python train_model.py --csv {os.path.basename(output_path)}")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _create_parser(dataset: str, data_dir: str, class_map_str: Optional[str] = None):
    """根据数据集类型创建解析器"""
    if dataset == "urfd":
        return URFDParser(data_dir)
    elif dataset == "le2i":
        return Le2iParser(data_dir)
    elif dataset == "sisfall":
        return SisFallParser(data_dir)
    elif dataset == "generic":
        class_map = parse_class_map(class_map_str or "")
        if not class_map:
            print("⚠ generic 模式建议提供 --class-map 参数")
            print("  示例: --class-map 'walk:walking,fall:fall,sit:sitting'")
        return GenericParser(data_dir, class_map)
    else:
        raise ValueError(f"未知数据集类型: {dataset}")


def _process_single_dataset(dataset: str, data_dir: str, output: str,
                            save_keypoints: Optional[str], stride: int,
                            max_frames: int, class_map_str: Optional[str] = None):
    """处理单个数据目录（被多目录模式调用）"""
    if not os.path.isdir(data_dir):
        print(f"  ⚠ 目录不存在，跳过: {data_dir}")
        return

    data_parser = _create_parser(dataset, data_dir, class_map_str)
    process_dataset(
        data_parser,
        output,
        save_keypoints_path=save_keypoints,
        feature_stride=stride,
        max_frames_per_video=max_frames,
    )


if __name__ == "__main__":
    main()
