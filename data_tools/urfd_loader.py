#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
urfd_loader.py — UR Fall Detection Dataset 加载与特征提取
===========================================================
将 URFD 的 PNG 序列 → MediaPipe 33关键点 → 空间运动特征 → 标注数据对

URFD 目录结构:
    URFD/
    ├── urfall-cam0-falls.csv          # 跌倒标注
    ├── urfall-cam0-adls.csv           # ADL 标注
    ├── Fall/
    │   ├── fall-01-cam0-rgb/          # PNG 序列
    │   ├── fall-01-cam0-depth/
    │   └── ...
    ├── ADL/
    │   ├── adl-01-cam0-rgb/
    │   └── ...
    └── README.txt

用法:
    conda activate fall
    python urfd_loader.py --data_dir E:/datasets/URFD --output E:/data/urfd_features/
"""

import argparse
import glob
import json
import os
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "src"))
import time
import re
from collections import defaultdict
import cv2
import numpy as np
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motion_spatial import MotionSpatialAnalyzer, MotionSpatial


class URFDLoader:
    """URFD 数据集加载器"""

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.frame_idx = 0
        self.prev_lm = None

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.motion_analyzer = MotionSpatialAnalyzer()

        self.fall_sequences = []
        self.adl_sequences = []
        self.total_frames = 0
        self.lost_frames = 0

    def discover_sequences(self) -> dict:
        """扫描目录，发现所有序列

        兼容两种 URFD 目录结构:
          1. 标准: {data_dir}/Fall/fall-*-cam0-rgb/ + ADL/adl-*-cam0-rgb/
          2. 扁平: {data_dir}/fall-*-cam0-rgb/ + adl-*-cam0-rgb/ (所有序列在同一层)
        """
        result = {"fall": [], "adl": []}

        for label, folder in [("fall", "Fall"), ("adl", "ADL")]:
            # 先尝试标准结构 (Fall/ + ADL/ 子目录)
            pattern = os.path.join(self.data_dir, folder, "*-cam0-rgb")
            seqs = sorted(glob.glob(pattern))

            # 如果标准结构没找到，尝试扁平结构 (直接在根目录下找)
            if not seqs:
                prefix = "fall" if label == "fall" else "adl"
                pattern = os.path.join(self.data_dir, f"{prefix}-*-cam0-rgb")
                seqs = sorted(glob.glob(pattern))

            for seq_path in seqs:
                name = os.path.basename(seq_path).replace("-cam0-rgb", "")
                images = sorted(glob.glob(os.path.join(seq_path, "*.png")))
                if images:
                    result[label].append({
                        "name": name,
                        "path": seq_path,
                        "images": images,
                        "label": 1 if label == "fall" else 0,
                    })

        print(f"  发现序列: 跌倒={len(result['fall'])} | ADL={len(result['adl'])}")
        total_imgs = sum(len(s["images"]) for v in result.values() for s in v)
        print(f"  总图片数: {total_imgs}")
        return result

    def load_annotations(self) -> dict:
        """加载 CSV 标注文件"""
        annotations = {}

        for label, csv_name in [("fall", "urfall-cam0-falls.csv"),
                                 ("adl", "urfall-cam0-adls.csv")]:
            csv_path = os.path.join(self.data_dir, csv_name)
            if not os.path.exists(csv_path):
                print(f"  [WARN] 标注文件不存在: {csv_path}")
                continue

            with open(csv_path, "r") as f:
                header = f.readline().strip().split(",")
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 4:
                        continue
                    # 常见格式: file, start, end, action, tag
                    seq_name = parts[0].replace("-cam0-rgb", "").strip()
                    start_frame = int(parts[1]) if parts[1].strip().isdigit() else 0
                    end_frame = int(parts[2]) if parts[2].strip().isdigit() else -1
                    action = parts[3].strip() if len(parts) > 3 else (
                        "fall" if label == "fall" else "adl")

                    annotations[seq_name] = {
                        "label": 1 if label == "fall" else 0,
                        "start_fall": start_frame,
                        "end_fall": end_frame,
                        "action": action,
                    }

        print(f"  加载标注: {len(annotations)} 条")
        return annotations

    def process_sequence(self, seq: dict, annotations: dict) -> dict:
        """处理单个序列：逐帧 MediaPipe + 特征提取"""
        name = seq["name"]
        images = seq["images"]
        label = seq["label"]
        ann = annotations.get(name, {})

        samples = []
        self.prev_lm = None
        frame_landmarks = []  # 存储每帧的 33 关键点

        for i, img_path in enumerate(images):
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            self.total_frames += 1

            if results.pose_landmarks:
                lm = self._extract_landmarks(results, frame.shape)
                frame_landmarks.append(lm)

                # 空间运动特征
                spatial = self.motion_analyzer.extract(lm, self.prev_lm)
                spatial_arr = spatial.to_array()

                # 躯干角度
                torso_angle = self._compute_torso_angle(lm)

                # 构造特征向量
                feat_vec = np.zeros(100, dtype=np.float32)
                feat_vec[1] = float(lm[23:25, 1].mean())  # centroid_y
                feat_vec[3] = torso_angle
                feat_vec[5] = float(lm[11:13, 1].mean())  # shoulder_mid_y

                # 帧级标注：根据标签和帧号判断
                frame_label = label
                if ann:
                    s, e = ann.get("start_fall", 0), ann.get("end_fall", -1)
                    if e > 0 and s <= i <= e:
                        frame_label = label

                samples.append({
                    "frame": i,
                    "features": feat_vec.tolist(),
                    "spatial": spatial_arr.tolist(),
                    "label": frame_label,
                    "torso_angle": torso_angle,
                    "landmarks": lm.tolist(),
                })

                self.prev_lm = lm
            else:
                self.lost_frames += 1
                # 用上一帧填充
                if self.prev_lm is not None and i > 0:
                    spatial = MotionSpatial(
                        motion_centroid_disp=0.0,
                        motion_centroid_angle=0.0,
                        motion_spread_active=0.0,
                        motion_spread_width=0.0,
                        torso_disp_ratio=0.0,
                        upper_lower_ratio=0.0,
                    )
                    spatial_arr = spatial.to_array()
                    torso_angle = self._compute_torso_angle(self.prev_lm)
                    feat_vec = np.zeros(100, dtype=np.float32)
                    feat_vec[1] = float(self.prev_lm[23:25, 1].mean())
                    feat_vec[3] = torso_angle
                    feat_vec[5] = float(self.prev_lm[11:13, 1].mean())

                    samples.append({
                        "frame": i,
                        "features": feat_vec.tolist(),
                        "spatial": spatial_arr.tolist(),
                        "label": label,
                        "torso_angle": torso_angle,
                        "landmarks": self.prev_lm.tolist(),
                        "interpolated": True,
                    })

        result = {
            "name": name,
            "label": label,
            "action": ann.get("action", "unknown"),
            "num_frames": len(images),
            "num_detected": len(frame_landmarks),
            "samples": samples,
        }

        if len(frame_landmarks) > 0:
            self._save_sequence_npz(result)

        return result

    def _save_sequence_npz(self, seq_result: dict):
        """保存序列为 .npz 格式（紧凑，快速加载）"""
        os.makedirs(self.output_dir, exist_ok=True)

        samples = seq_result["samples"]
        n = len(samples)
        if n == 0:
            return

        # 打包为数组
        features = np.array([s["features"] for s in samples], dtype=np.float32)
        spatial = np.array([s["spatial"] for s in samples], dtype=np.float32)
        labels = np.array([s["label"] for s in samples], dtype=np.int8)
        torso_angles = np.array([s["torso_angle"] for s in samples], dtype=np.float32)
        landmarks = np.array([s.get("landmarks",
                             np.zeros((33, 4)).tolist()) for s in samples], dtype=np.float32)

        path = os.path.join(self.output_dir, f"{seq_result['name']}.npz")
        np.savez_compressed(path,
                            features=features,
                            spatial=spatial,
                            labels=labels,
                            torso_angles=torso_angles,
                            landmarks=landmarks,
                            label=seq_result["label"],
                            action=seq_result["action"])

    def _extract_landmarks(self, results, frame_shape) -> np.ndarray:
        """从 MediaPipe results 提取 (33, 4) 关键点"""
        h, w = frame_shape[:2]
        lm = np.zeros((33, 4), dtype=np.float32)
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            lm[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
        return lm

    @staticmethod
    def _compute_torso_angle(lm: np.ndarray) -> float:
        shoulder_mid = (lm[11, :2] + lm[12, :2]) / 2
        hip_mid = (lm[23, :2] + lm[24, :2]) / 2
        dy = hip_mid[1] - shoulder_mid[1]
        dx = hip_mid[0] - shoulder_mid[0]
        return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))

    def run(self) -> dict:
        """主流程：发现 → 标注 → 处理 → 保存摘要"""
        print("=" * 60)
        print("  URFD 数据加载与特征提取")
        print("=" * 60)

        sequences = self.discover_sequences()
        annotations = self.load_annotations()

        summary = {"fall": [], "adl": [], "stats": {}}
        total_processed = 0

        for label in ["fall", "adl"]:
            print(f"\n  处理 {label.upper()} 序列 ({len(sequences[label])} 个)...")
            for seq in sequences[label]:
                name = seq["name"]
                start_t = time.time()
                result = self.process_sequence(seq, annotations)
                elapsed = time.time() - start_t
                summary[label].append({
                    "name": result["name"],
                    "label": result["label"],
                    "action": result["action"],
                    "frames": result["num_frames"],
                    "detected": result["num_detected"],
                })
                total_processed += 1
                if total_processed % 10 == 0:
                    print(f"    [{total_processed}] {name} ({elapsed:.1f}s)")

        # 统计
        summary["stats"] = {
            "fall_sequences": len(sequences["fall"]),
            "adl_sequences": len(sequences["adl"]),
            "total_frames": self.total_frames,
            "lost_frames": self.lost_frames,
            "detection_rate": round(100 * (1 - self.lost_frames / max(self.total_frames, 1)), 1),
            "output_dir": self.output_dir,
        }

        # 保存摘要
        summary_path = os.path.join(self.output_dir, "dataset_summary.json")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 60}")
        print(f"  摘要: {summary['stats']}")
        print(f"  输出: {self.output_dir}")
        print(f"  检测率: {summary['stats']['detection_rate']}%")
        print(f"  摘要文件: {summary_path}")
        print(f"{'=' * 60}")

        self.pose.close()
        return summary


def main():
    parser = argparse.ArgumentParser(description="URFD 加载 & 特征提取")
    parser.add_argument("--data_dir", "-d", type=str, required=True,
                        help="URFD 数据集根目录")
    parser.add_argument("--output", "-o", type=str, default="E:/老人跌倒/data/urfd_features/",
                        help="特征输出目录")
    args = parser.parse_args()

    loader = URFDLoader(args.data_dir, args.output)
    loader.run()


if __name__ == "__main__":
    main()
