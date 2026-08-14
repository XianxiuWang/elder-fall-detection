#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_fall_classifier.py — 跌倒 ML 分类器训练（Phase A）
=========================================================
从 URFD .npz 特征 + 模拟场景数据训练 LightGBM / XGBoost / RF 分类器。

架构:
  滑动窗口 → 多维度统计特征(30+维) → 二分类(跌倒/正常)
  输出: ML 概率 + 硬阈值 → 双重判定

用法:
    # 用 URFD 数据训练
    python train_fall_classifier.py --data_dir E:/老人跌倒/data/urfd_features/

    # 用模拟场景数据训练（快速验证）
    python train_fall_classifier.py --synthetic

    # 全模式训练 + 对比
    python train_fall_classifier.py --all

    # 输出模型到指定路径
    python train_fall_classifier.py --synthetic --model_out E:/老人跌倒/models/fall_lgb.pkl
"""

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np

warnings.filterwarnings("ignore")

# ============================================================
# 配置
# ============================================================

@dataclass
class TrainConfig:
    """训练配置"""
    window_size: int = 30          # 滑动窗口帧数（30fps → 1秒）
    window_stride: int = 5         # 窗口滑动步长
    test_ratio: float = 0.2        # 验证集比例
    cv_folds: int = 5              # 交叉验证折数
    random_state: int = 42
    model_type: str = "lgb"        # lgb / xgb / rf
    top_n_features: int = 20       # 最终保留最重要特征数
    fall_threshold: float = 0.6    # ML 判定阈值

    # LightGBM 超参
    lgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 6,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "class_weight": "balanced",
    })


# ============================================================
# 特征工程
# ============================================================

# MediaPipe 33关键点索引
NOSE, LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER = 0, 1, 2, 3
RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER = 4, 5, 6
LEFT_EAR, RIGHT_EAR = 7, 8
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

SHOULDER_MID = None  # (11 + 12) / 2
HIP_MID = None       # (23 + 24) / 2
KNEE_MID = None      # (25 + 26) / 2


class FeatureExtractor:
    """从关键点序列提取窗口级特征"""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size

    def extract_window(self, landmarks_seq: np.ndarray) -> np.ndarray:
        """
        从一个窗口的关键点序列提取特征向量。
        landmarks_seq: (window_size, 33, 3) 或 (window_size, 33, 4)
        返回: (n_features,) 特征向量
        """
        if len(landmarks_seq) < 3:
            return np.zeros(33, dtype=np.float32)

        lm = landmarks_seq.copy()  # (T, 33, >=3)
        T = len(lm)
        x = lm[:, :, 0]  # (T, 33)
        y = lm[:, :, 1]  # (T, 33)
        z = lm[:, :, 2] if lm.shape[2] >= 3 else np.zeros_like(x)

        # 可见性（如果有）
        vis = lm[:, :, 3] if lm.shape[2] >= 4 else np.ones_like(x)

        features = {}

        # ---- 1. 头部轨迹特征 (nose) ----
        nose_y = y[:, NOSE]
        nose_vy = np.gradient(nose_y)
        features["head_y_min"] = float(np.min(nose_y))
        features["head_y_max"] = float(np.max(nose_y))
        features["head_y_range"] = float(np.ptp(nose_y))
        features["head_y_drop"] = float(nose_y[0] - nose_y[-1])      # 正=降低
        features["head_vy_max"] = float(np.max(np.abs(nose_vy)))
        features["head_vy_mean"] = float(np.mean(np.abs(nose_vy)))

        # ---- 2. 躯干特征 ----
        shoulder_mid_y = (y[:, LEFT_SHOULDER] + y[:, RIGHT_SHOULDER]) / 2
        hip_mid_y = (y[:, LEFT_HIP] + y[:, RIGHT_HIP]) / 2
        torso_center_y = (shoulder_mid_y + hip_mid_y) / 2

        features["torso_y_drop"] = float(torso_center_y[0] - torso_center_y[-1])
        features["torso_y_min"] = float(np.min(torso_center_y))
        features["torso_y_range"] = float(np.ptp(torso_center_y))

        # 躯干角度（与垂直线的夹角）
        shoulder_mid_x = (x[:, LEFT_SHOULDER] + x[:, RIGHT_SHOULDER]) / 2
        hip_mid_x = (x[:, LEFT_HIP] + x[:, RIGHT_HIP]) / 2
        dx = hip_mid_x - shoulder_mid_x
        dy = hip_mid_y - shoulder_mid_y + 1e-6
        torso_angles = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
        features["torso_angle_max"] = float(np.max(torso_angles))
        features["torso_angle_mean"] = float(np.mean(torso_angles))
        features["torso_angle_final"] = float(torso_angles[-1]) if len(torso_angles) > 0 else 0
        features["torso_angle_change"] = float(np.ptp(torso_angles))

        # ---- 3. 运动中心位移 ----
        all_y_mid = np.mean(y[:, [LEFT_SHOULDER, RIGHT_SHOULDER,
                                   LEFT_HIP, RIGHT_HIP,
                                   LEFT_KNEE, RIGHT_KNEE]], axis=1)
        centroid_disp = np.sqrt(
            np.diff(all_y_mid, prepend=all_y_mid[0:1]) ** 2
        )
        features["centroid_disp_max"] = float(np.max(centroid_disp))
        features["centroid_disp_mean"] = float(np.mean(centroid_disp))
        features["centroid_disp_std"] = float(np.std(centroid_disp))
        features["centroid_total_disp"] = float(all_y_mid[-1] - all_y_mid[0])

        # ---- 4. 运动扩散范围 (spread) ----
        active_y_range = np.ptp(y, axis=1)  # 每帧所有点的 y 跨度
        features["spread_max"] = float(np.max(active_y_range))
        features["spread_mean"] = float(np.mean(active_y_range))
        features["spread_std"] = float(np.std(active_y_range))

        # ---- 5. 速度特征 ----
        centroid_vy = np.abs(np.gradient(all_y_mid))
        features["speed_max"] = float(np.max(centroid_vy))
        features["speed_mean"] = float(np.mean(centroid_vy))
        features["speed_std"] = float(np.std(centroid_vy))

        # 达到最大速度的时间点（比率）
        if features["speed_max"] > 0:
            peak_idx = np.argmax(centroid_vy)
            features["speed_peak_position"] = float(peak_idx / max(T, 1))
        else:
            features["speed_peak_position"] = 0.5

        # 加速度（速度的导数）
        accel = np.abs(np.gradient(centroid_vy))
        features["accel_max"] = float(np.max(accel))
        features["accel_mean"] = float(np.mean(accel))

        # ---- 6. 静止恢复特征（窗末尾几帧） ----
        tail_ratio = 0.25
        tail_n = max(1, int(T * tail_ratio))
        tail_frames = centroid_vy[-tail_n:]
        head_frames = centroid_vy[:tail_n]
        features["stillness_tail"] = float(np.mean(tail_frames))
        features["speed_ratio_tail_head"] = (
            float(np.mean(tail_frames) / (np.mean(head_frames) + 1e-6))
        )

        # ---- 7. 下肢特征 ----
        knee_mid_y = (y[:, LEFT_KNEE] + y[:, RIGHT_KNEE]) / 2
        ankle_mid_y = (y[:, LEFT_ANKLE] + y[:, RIGHT_ANKLE]) / 2
        features["knee_y_drop"] = float(knee_mid_y[0] - knee_mid_y[-1])
        features["ankle_y_range"] = float(np.ptp(ankle_mid_y))

        # 髋膝踝垂直分布（跌倒时三点更接近地面）
        features["hip_knee_ankle_spread_start"] = float(
            np.ptp([hip_mid_y[0], knee_mid_y[0], ankle_mid_y[0]])
        )
        features["hip_knee_ankle_spread_end"] = float(
            np.ptp([hip_mid_y[-1], knee_mid_y[-1], ankle_mid_y[-1]])
        )
        features["hip_knee_ankle_spread_min"] = float(
            min(np.min(hip_mid_y), np.min(knee_mid_y), np.min(ankle_mid_y))
        )

        # ---- 8. 上肢活动度 ----
        wrist_y_mean = (y[:, LEFT_WRIST] + y[:, RIGHT_WRIST]) / 2
        elbow_y_mean = (y[:, LEFT_ELBOW] + y[:, RIGHT_ELBOW]) / 2
        features["wrist_range"] = float(np.ptp(wrist_y_mean))
        features["elbow_wrist_dist"] = float(
            np.mean(np.abs(wrist_y_mean - elbow_y_mean))
        )

        # ---- 9. 躯干-头部偏差 ----
        features["head_torso_offset"] = float(
            np.mean(np.abs(nose_y - torso_center_y))
        )

        # ---- 10. 距离域特征 (雷达论文启发: "速度是表象, 距离变化才是结构信号") ----
        # 速度加权重心：运动能量集中在身体哪个位置
        # 跌倒时重心快速下移, 日常动作重心变化小
        centroid_vy_all = np.abs(np.gradient(all_y_mid))
        speed_weights = centroid_vy_all / (np.sum(centroid_vy_all) + 1e-6)
        features["motion_weighted_centroid"] = float(
            np.sum(speed_weights * all_y_mid)
        )

        # 活跃身体跨度：窗口首尾帧的身体Y跨度
        body_y_span = np.ptp(y, axis=1)  # 每帧所有关键点的Y范围
        start_n = min(5, T)
        end_n = min(5, T)
        features["active_range_start"] = float(np.mean(body_y_span[:start_n]))
        features["active_range_end"] = float(np.mean(body_y_span[-end_n:]))
        # 跨度比: 跌倒时身体收缩→比值变小
        features["active_range_ratio"] = float(
            features["active_range_end"] / (features["active_range_start"] + 1e-6)
        )

        # 速度-距离比: 高速+小位移=局部动作(日常), 高速+大位移=全身动作(跌倒)
        total_displacement = float(np.abs(all_y_mid[-1] - all_y_mid[0]))
        features["speed_distance_ratio"] = float(
            features["speed_max"] / (total_displacement + 1e-6)
        )

        # 身体紧凑度: 跌倒时身体收缩成团→均值/最大值变小
        features["body_compactness"] = float(
            np.mean(body_y_span) / (np.max(body_y_span) + 1e-6)
        )

        return np.array(list(features.values()), dtype=np.float32), list(features.keys())

    def sliding_windows(self, landmarks_seq: np.ndarray,
                        stride: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """滑动窗口提取特征序列"""
        if len(landmarks_seq) < self.window_size:
            return np.zeros((0, 1)), np.zeros(0)

        X_list, feat_names = [], None
        for start in range(0, len(landmarks_seq) - self.window_size + 1, stride):
            window = landmarks_seq[start:start + self.window_size]
            vec, names = self.extract_window(window)
            X_list.append(vec)
            feat_names = names

        return np.array(X_list), np.array(feat_names) if feat_names else np.array([])


# ============================================================
# 数据生成（模拟场景）
# ============================================================

class SyntheticDataGenerator:
    """生成模拟关键点数据用于快速验证"""

    def __init__(self, n_frames_per_seq: int = 60, n_sequences_per_class: int = 200):
        self.n_frames = n_frames_per_seq
        self.n_seqs = n_sequences_per_class

    def _base_keypoints(self) -> np.ndarray:
        """站立姿态 (33, 3)"""
        kp = np.zeros((33, 3))
        # 简化站立骨架
        kp[:, 0] = 0.5  # x 居中
        kp[:, 2] = 0.0  # z 平面

        # 头部
        kp[0, 1] = 0.15     # nose
        kp[1:9, 1] = 0.16   # eyes+ears+mouth
        # 肩膀
        kp[11, 1] = 0.28; kp[11, 0] = 0.42
        kp[12, 1] = 0.28; kp[12, 0] = 0.58
        # 肘
        kp[13, 1] = 0.42; kp[13, 0] = 0.35
        kp[14, 1] = 0.42; kp[14, 0] = 0.65
        # 腕
        kp[15, 1] = 0.55; kp[15, 0] = 0.30
        kp[16, 1] = 0.55; kp[16, 0] = 0.70
        # 髋
        kp[23, 1] = 0.48; kp[23, 0] = 0.43
        kp[24, 1] = 0.48; kp[24, 0] = 0.57
        # 膝
        kp[25, 1] = 0.68; kp[25, 0] = 0.43
        kp[26, 1] = 0.68; kp[26, 0] = 0.57
        # 踝
        kp[27, 1] = 0.88; kp[27, 0] = 0.43
        kp[28, 1] = 0.88; kp[28, 0] = 0.57
        # 脚
        kp[29, 1] = 0.92; kp[29, 0] = 0.43
        kp[30, 1] = 0.92; kp[30, 0] = 0.57
        kp[31, 1] = 0.95; kp[31, 0] = 0.43
        kp[32, 1] = 0.95; kp[32, 0] = 0.57

        return kp

    def generate_fall_sequence(self, noise_level=0.005) -> np.ndarray:
        """生成跌倒序列：站立 → 快速向下"""
        base = self._base_keypoints()
        seq = np.tile(base[np.newaxis, :, :], (self.n_frames, 1, 1))

        # 跌倒点：30-50帧开始
        fall_start = np.random.randint(20, 35)
        fall_end = min(fall_start + np.random.randint(10, 20), self.n_frames - 5)

        # 快速下落
        for t in range(fall_start, fall_end):
            progress = (t - fall_start) / max(fall_end - fall_start, 1)
            drop = 0.4 * (progress ** 2)  # 加速下落
            # 所有下半身点下移
            lower_idxs = list(range(23, 33))  # 髋以下
            seq[t, lower_idxs, 1] += drop

            # 躯干倾斜
            lean = 0.15 * progress
            seq[t, 11:25, 0] += lean * (1 if np.random.random() > 0.5 else -1)

        # 跌倒后保持低姿态
        final_drop = 0.4
        for t in range(fall_end, self.n_frames):
            seq[t, list(range(23, 33)), 1] += final_drop

        # 噪声
        seq += np.random.normal(0, noise_level, seq.shape)
        return seq

    def generate_walking_sequence(self, noise_level=0.005) -> np.ndarray:
        """生成行走序列：周期性摆动"""
        base = self._base_keypoints()
        seq = np.tile(base[np.newaxis, :, :], (self.n_frames, 1, 1))

        for t in range(self.n_frames):
            # 小幅度周期性摆动
            swing = 0.02 * np.sin(t * 0.3)
            seq[t, 25:29, 1] += swing  # 膝踝上下
            seq[t, 15, 1] += swing * 0.5  # 手腕
            seq[t, 16, 1] -= swing * 0.5

            # 缓慢移动
            walk = 0.003 * t
            seq[t, :, 0] += walk * 0.3

        seq += np.random.normal(0, noise_level, seq.shape)
        return seq

    def generate_bending_sequence(self, noise_level=0.005) -> np.ndarray:
        """弯腰：躯干前倾但不下落"""
        base = self._base_keypoints()
        seq = np.tile(base[np.newaxis, :, :], (self.n_frames, 1, 1))

        bend_start = np.random.randint(15, 25)
        bend_mid = bend_start + np.random.randint(8, 15)
        bend_end = bend_mid + np.random.randint(8, 15)

        # 弯腰（上半身下移，髋部几乎不动）
        for t in range(bend_start, bend_mid):
            progress = (t - bend_start) / max(bend_mid - bend_start, 1)
            drop = 0.25 * progress
            seq[t, :13, 1] += drop  # 上半身
            seq[t, 13:17, 1] += drop * 0.8  # 肘腕

        # 回复
        for t in range(bend_mid, min(bend_end, self.n_frames)):
            progress = (t - bend_mid) / max(bend_end - bend_mid, 1)
            seq[t, :17, 1] -= 0.25 * (1 - progress)

        seq += np.random.normal(0, noise_level, seq.shape)
        return seq

    def generate_sitting_sequence(self, noise_level=0.005) -> np.ndarray:
        """坐下：缓慢向下 + 膝弯曲"""
        base = self._base_keypoints()
        seq = np.tile(base[np.newaxis, :, :], (self.n_frames, 1, 1))

        sit_start = np.random.randint(20, 30)
        sit_duration = np.random.randint(15, 25)

        for t in range(sit_start, min(sit_start + sit_duration, self.n_frames)):
            progress = (t - sit_start) / sit_duration
            drop = 0.3 * progress  # 缓慢下落

            # 整体下移 + 膝弯曲
            body = list(range(11, 33))
            seq[t, body, 1] += drop
            seq[t, [25, 26], 1] += progress * 0.15  # 膝弯曲更多

        # 坐定后保持
        final_t = min(sit_start + sit_duration, self.n_frames)
        for t in range(final_t, self.n_frames):
            seq[t, list(range(11, 33)), 1] += 0.3

        seq += np.random.normal(0, noise_level, seq.shape)
        return seq

    def generate_waving_sequence(self, noise_level=0.005) -> np.ndarray:
        """挥手：上肢动，躯干不动"""
        base = self._base_keypoints()
        seq = np.tile(base[np.newaxis, :, :], (self.n_frames, 1, 1))

        for t in range(self.n_frames):
            wave = 0.08 * np.sin(t * 0.5)
            seq[t, 15, 1] += wave  # 右手腕
            seq[t, 13, 1] += wave * 0.3  # 右肘
            seq[t, 14, 1] += wave * 0.2  # 左肘

        seq += np.random.normal(0, noise_level, seq.shape)
        return seq

    def generate_squatting_sequence(self, noise_level=0.005) -> np.ndarray:
        """蹲下：慢速 + 膝深屈"""
        base = self._base_keypoints()
        seq = np.tile(base[np.newaxis, :, :], (self.n_frames, 1, 1))

        squat_start = np.random.randint(20, 30)
        for t in range(squat_start, self.n_frames):
            progress = min((t - squat_start) / 20.0, 1.0)
            drop = 0.05 * progress
            seq[t, list(range(11, 33)), 1] += drop
            seq[t, [25, 26], 1] += progress * 0.2
            seq[t, [27, 28], 1] += progress * 0.05  # 脚跟微抬

        seq += np.random.normal(0, noise_level, seq.shape)
        return seq

    def generate_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成全部数据"""
        X_list, y_list = [], []

        generators = [
            (self.generate_fall_sequence, 1),
            (self.generate_walking_sequence, 0),
            (self.generate_bending_sequence, 0),
            (self.generate_sitting_sequence, 0),
            (self.generate_waving_sequence, 0),
            (self.generate_squatting_sequence, 0),
        ]

        for gen_fn, label in generators:
            print(f"  生成 {gen_fn.__name__} ({label}) x {self.n_seqs}")
            for _ in range(self.n_seqs):
                seq = gen_fn()
                X_list.append(seq)
                y_list.append(label)

        return np.array(X_list), np.array(y_list)


# ============================================================
# 数据加载
# ============================================================

def load_urfd_features(data_dir: str) -> Tuple[List[np.ndarray], List[int]]:
    """加载 URFD .npz 特征文件"""
    sequences, labels = [], []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz") or fname == "metadata.npz":
            continue
        path = os.path.join(data_dir, fname)
        try:
            data = np.load(path, allow_pickle=True)
            if "landmarks" in data:
                lm = data["landmarks"]
                if lm.ndim == 3 and lm.shape[1] == 33:
                    sequences.append(lm)
                    # 从文件读取标签
                    label = int(data.get("label", 0))
                    labels.append(label)
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")

    print(f"  加载 URFD: {len(sequences)} 序列")
    return sequences, labels


def load_synthetic_data() -> Tuple[List[np.ndarray], List[int]]:
    """生成模拟数据"""
    print("  生成模拟训练数据...")
    gen = SyntheticDataGenerator(n_frames_per_seq=60, n_sequences_per_class=200)
    seqs, labels = gen.generate_all()
    print(f"  序列: {len(seqs)}, 标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    # seqs: (n, T, 33, 3) → list of (T, 33, 3)
    return list(seqs), list(labels)


# ============================================================
# 训练器
# ============================================================

class FallClassifierTrainer:
    """跌倒分类器训练"""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.extractor = FeatureExtractor(window_size=config.window_size)
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.top_indices = None
        self.metrics = {}

    def prepare_dataset(self, sequences: List[np.ndarray],
                        labels: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """提取窗口特征 + 标签"""
        X_all, y_all = [], []

        for seq, label in zip(sequences, labels):
            X_windows, feat_names = self.extractor.sliding_windows(
                seq, stride=self.config.window_stride
            )
            if X_windows.shape[0] > 0:
                X_all.append(X_windows)
                y_all.extend([label] * len(X_windows))

        if len(X_all) == 0:
            raise ValueError("没有提取到任何特征窗口！")

        X = np.vstack(X_all).astype(np.float32)
        y = np.array(y_all, dtype=np.int32)
        self.feature_names = feat_names

        # 处理 NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"  特征矩阵: {X.shape}, 标签: pos={sum(y)}, neg={len(y)-sum(y)} "
              f"({100*sum(y)/len(y):.1f}% 正样本)")
        return X, y, feat_names

    def train_val_split(self, X, y):
        """分层划分训练/验证集"""
        from sklearn.model_selection import train_test_split
        return train_test_split(
            X, y, test_size=self.config.test_ratio,
            stratify=y, random_state=self.config.random_state
        )

    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """训练 LightGBM"""
        from sklearn.preprocessing import StandardScaler
        import lightgbm as lgb

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        model = lgb.LGBMClassifier(**self.config.lgb_params,
                                    random_state=self.config.random_state,
                                    verbose=-1)
        model.fit(X_train_s, y_train,
                  eval_set=[(X_val_s, y_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=30),
                             lgb.log_evaluation(period=0)])

        return model, X_train_s, X_val_s

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """训练 XGBoost"""
        from sklearn.preprocessing import StandardScaler
        import xgboost as xgb

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        # 计算 scale_pos_weight
        n_pos = sum(y_train)
        n_neg = len(y_train) - n_pos
        scale_weight = n_neg / max(n_pos, 1)

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_weight,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=self.config.random_state,
            verbosity=0,
        )
        model.fit(X_train_s, y_train,
                  eval_set=[(X_val_s, y_val)], verbose=False)

        return model, X_train_s, X_val_s

    def train_randomforest(self, X_train, y_train, X_val, y_val):
        """训练 RandomForest"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val)

        model = RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced",
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        model.fit(X_train_s, y_train)

        return model, X_train_s, X_val_s

    def cross_validate(self, X, y) -> dict:
        """K 折交叉验证"""
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import StandardScaler

        cv = StratifiedKFold(n_splits=self.config.cv_folds,
                             shuffle=True, random_state=self.config.random_state)

        # 标准化
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        # 使用 LightGBM
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**self.config.lgb_params,
                                    random_state=self.config.random_state,
                                    verbose=-1)

        scores = {}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            try:
                s = cross_val_score(model, X_s, y, cv=cv, scoring=metric, n_jobs=-1)
                scores[metric] = float(np.mean(s))
                scores[f"{metric}_std"] = float(np.std(s))
            except Exception:
                scores[metric] = 0.0

        return scores

    def evaluate(self, model, X_val, y_val) -> dict:
        """详细评估"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix,
            classification_report,
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= self.config.fall_threshold).astype(int)

        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        return {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, y_prob)),
            "confusion_matrix": {"TP": int(tp), "FP": int(fp),
                                  "TN": int(tn), "FN": int(fn)},
            "threshold": self.config.fall_threshold,
        }

    def feature_importance(self, model, top_n: int = 20) -> List[Dict]:
        """特征重要性排序"""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "get_booster"):
            importances = model.get_booster().get_score(importance_type="gain")
            # 对齐到特征名
            imp_arr = np.zeros(len(self.feature_names))
            for k, v in importances.items():
                try:
                    idx = int(k.replace("f", ""))
                    if idx < len(imp_arr):
                        imp_arr[idx] = v
                except ValueError:
                    pass
            importances = imp_arr
        else:
            return []

        # 排序
        ranked = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1], reverse=True
        )[:top_n]

        return [{"feature": f, "importance": float(i)} for f, i in ranked]

    def run(self, sequences: List[np.ndarray], labels: List[int]) -> dict:
        """完整训练流程"""
        print(f"\n{'=' * 60}")
        print(f"  Phase A: ML 跌倒分类器训练")
        print(f"{'=' * 60}")

        # ---- 1. 特征提取 ----
        print("\n[1/5] 滑动窗口特征提取...")
        X, y, feat_names = self.prepare_dataset(sequences, labels)

        # ---- 2. 交叉验证 ----
        print("\n[2/5] 交叉验证...")
        cv_scores = self.cross_validate(X, y)
        print(f"  CV Accuracy:  {cv_scores.get('accuracy', 0):.4f} ± {cv_scores.get('accuracy_std', 0):.4f}")
        print(f"  CV F1:        {cv_scores.get('f1', 0):.4f} ± {cv_scores.get('f1_std', 0):.4f}")
        print(f"  CV Recall:    {cv_scores.get('recall', 0):.4f} ± {cv_scores.get('recall_std', 0):.4f}")
        print(f"  CV Precision: {cv_scores.get('precision', 0):.4f} ± {cv_scores.get('precision_std', 0):.4f}")

        # ---- 3. 划分 & 训练 ----
        print(f"\n[3/5] 训练 {self.config.model_type.upper()} 模型...")
        X_train, X_val, y_train, y_val = self.train_val_split(X, y)

        t0 = time.time()
        if self.config.model_type == "lgb":
            self.model, X_train_s, X_val_s = self.train_lightgbm(
                X_train, y_train, X_val, y_val
            )
        elif self.config.model_type == "xgb":
            self.model, X_train_s, X_val_s = self.train_xgboost(
                X_train, y_train, X_val, y_val
            )
        elif self.config.model_type == "rf":
            self.model, X_train_s, X_val_s = self.train_randomforest(
                X_train, y_train, X_val, y_val
            )
        else:
            raise ValueError(f"未知模型类型: {self.config.model_type}")

        train_time = time.time() - t0
        print(f"  训练耗时: {train_time:.1f}s")

        # ---- 4. 评估 ----
        print("\n[4/5] 模型评估...")
        eval_metrics = self.evaluate(self.model, X_val_s, y_val)
        print(f"  Accuracy:   {eval_metrics['accuracy']:.4f}")
        print(f"  Precision:  {eval_metrics['precision']:.4f}")
        print(f"  Recall:     {eval_metrics['recall']:.4f}")
        print(f"  F1 Score:   {eval_metrics['f1']:.4f}")
        print(f"  ROC-AUC:    {eval_metrics['roc_auc']:.4f}")
        cm = eval_metrics["confusion_matrix"]
        print(f"  Confusion:  TP={cm['TP']} FP={cm['FP']} TN={cm['TN']} FN={cm['FN']}")

        # ---- 5. 特征重要性 & 精简 ----
        print("\n[5/5] 特征重要性分析...")
        fi = self.feature_importance(self.model, top_n=self.config.top_n_features)
        print(f"\n  Top-{self.config.top_n_features} 重要特征:")
        for i, item in enumerate(fi[:10]):
            bar = "█" * int(item["importance"] * 40 / max(fi[0]["importance"], 1e-6))
            print(f"    {i+1:2d}. {item['feature']:<35s} {item['importance']:.4f} {bar}")

        # 保存到特征名
        self.top_indices = [list(feat_names).index(f["feature"])
                            for f in fi[:self.config.top_n_features]
                            if f["feature"] in feat_names]
        print(f"\n  精选 {len(self.top_indices)} 维特征用于部署")

        # ---- 汇总 ----
        self.metrics = {
            "cv": cv_scores,
            "eval": eval_metrics,
            "feature_importance": fi,
            "train_time_s": train_time,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "top_n_features": len(self.top_indices),
            "model_type": self.config.model_type,
            "window_size": self.config.window_size,
            "window_stride": self.config.window_stride,
        }

        return self.metrics

    def compare_models(self, sequences, labels):
        """对比 3 种模型"""
        print(f"\n{'=' * 60}")
        print(f"  多模型对比 (LightGBM vs XGBoost vs RandomForest)")
        print(f"{'=' * 60}")

        X, y, _ = self.prepare_dataset(sequences, labels)
        X_train, X_val, y_train, y_val = self.train_val_split(X, y)

        results = {}
        for name, train_fn in [("LightGBM", self.train_lightgbm),
                                ("XGBoost", self.train_xgboost),
                                ("RandomForest", self.train_randomforest)]:
            print(f"\n  --- {name} ---")
            t0 = time.time()
            model, X_tr_s, X_vl_s = train_fn(X_train, y_train, X_val, y_val)
            elapsed = time.time() - t0
            metrics = self.evaluate(model, X_vl_s, y_val)
            fi = self.feature_importance(model, top_n=10)
            results[name] = {**metrics, "time_s": elapsed,
                             "top_features": [f["feature"] for f in fi[:5]]}

            print(f"    F1={metrics['f1']:.4f}  "
                  f"Recall={metrics['recall']:.4f}  "
                  f"Precision={metrics['precision']:.4f}  "
                  f"AUC={metrics['roc_auc']:.4f}  "
                  f"Time={elapsed:.1f}s")

        # 最佳模型
        best = max(results, key=lambda k: results[k]["f1"])
        print(f"\n  ★ 最佳模型: {best} (F1={results[best]['f1']:.4f})")

        self.config.model_type = {"LightGBM": "lgb", "XGBoost": "xgb",
                                   "RandomForest": "rf"}[best]
        # 用最佳模型重新训练
        if self.config.model_type == "lgb":
            self.model, _, _ = self.train_lightgbm(X_train, y_train, X_val, y_val)
        elif self.config.model_type == "xgb":
            self.model, _, _ = self.train_xgboost(X_train, y_train, X_val, y_val)
        else:
            self.model, _, _ = self.train_randomforest(X_train, y_train, X_val, y_val)

        return results

    def save_model(self, path: str):
        """保存模型 + 元数据"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        bundle = {
            "model": self.model,
            "scaler": self.scaler,
            "config": self.config,
            "metrics": self.metrics,
            "feature_names": list(self.feature_names) if self.feature_names is not None else [],
            "top_indices": self.top_indices,
        }

        with open(path, "wb") as f:
            pickle.dump(bundle, f)

        # 同时保存 JSON 报告
        report_path = path.replace(".pkl", "_report.json")
        report = {
            k: v for k, v in self.metrics.items()
            if k not in ["feature_importance"]
        }
        if "feature_importance" in self.metrics:
            report["top_10_features"] = self.metrics["feature_importance"][:10]
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n  [OK] 模型保存: {path}")
        print(f"  [OK] 报告保存: {report_path}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="跌倒 ML 分类器训练 (Phase A)")
    parser.add_argument("--data_dir", "-d", type=str, default=None,
                        help="URFD 特征数据目录")
    parser.add_argument("--synthetic", "-s", action="store_true",
                        help="使用模拟数据训练")
    parser.add_argument("--all", "-a", action="store_true",
                        help="对比所有模型")
    parser.add_argument("--model_out", "-o", type=str,
                        default="E:/老人跌倒/models/fall_classifier.pkl",
                        help="模型输出路径")
    parser.add_argument("--model_type", "-m", type=str, default="lgb",
                        choices=["lgb", "xgb", "rf"],
                        help="模型类型")
    parser.add_argument("--window_size", "-w", type=int, default=30,
                        help="滑动窗口帧数")
    parser.add_argument("--compare", action="store_true",
                        help="多模型对比模式")
    args = parser.parse_args()

    # ---- 加载数据 ----
    sequences, labels = [], []

    if args.synthetic:
        seq_s, lbl_s = load_synthetic_data()
        sequences.extend(seq_s)
        labels.extend(lbl_s)

    if args.data_dir:
        seq_u, lbl_u = load_urfd_features(args.data_dir)
        sequences.extend(seq_u)
        labels.extend(lbl_u)

    if not sequences:
        print("[ERROR] 没有数据！请指定 --synthetic 或 --data_dir")
        print("  示例: python train_fall_classifier.py --synthetic")
        sys.exit(1)

    # ---- 训练 ----
    config = TrainConfig(
        window_size=args.window_size,
        model_type=args.model_type,
    )

    trainer = FallClassifierTrainer(config)

    if args.compare or args.all:
        results = trainer.compare_models(sequences, labels)
    else:
        metrics = trainer.run(sequences, labels)

    # ---- 保存 ----
    trainer.save_model(args.model_out)

    print(f"\n{'=' * 60}")
    print(f"  Phase A 训练完成！")
    print(f"  模型: {args.model_out}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
