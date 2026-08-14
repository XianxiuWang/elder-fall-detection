#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_fall_detector.py — ML 跌倒推理模块
=======================================
加载训练好的 LightGBM 模型，对实时关键点流做滑动窗口推理。

与 e2e_fall_monitor.py 集成方式:
    detector = MLFallDetector("E:/老人跌倒/models/fall_classifier.pkl")
    prob = detector.update(landmarks_33x4)  # 每帧调用
    if prob >= 0.7:
        print("跌倒!")

用法:
    # 独立测试
    python ml_fall_detector.py --model E:/老人跌倒/models/fall_classifier.pkl --test
"""

import argparse
import os
import pickle
import sys
import time
from typing import Optional, Tuple
from collections import deque
import numpy as np

# 加载 TrainConfig 以备 pickle 反序列化（训练脚本作为 __main__ 运行时保存）
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_proj_root, 'training'))
try:
    from train_fall_classifier import TrainConfig
except ImportError:
    TrainConfig = type('TrainConfig', (), {})  # dummy fallback
import __main__ as _main
_main.TrainConfig = TrainConfig


class MLFallDetector:
    """基于 ML 的跌倒检测推理器"""

    def __init__(self, model_path: str,
                 window_size: int = 30,
                 stride: int = 5,
                 threshold: float = 0.6):
        """
        Args:
            model_path: 训练好的模型 pickle 路径
            window_size: 滑动窗口帧数
            stride: 特征提取步长
            threshold: 判定为跌倒的概率阈值
        """
        self.window_size = window_size
        self.stride = stride
        self.threshold = threshold

        # 加载模型
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_names = bundle.get("feature_names", [])
        self.top_indices = bundle.get("top_indices", None)
        self.metrics = bundle.get("metrics", {})
        self.train_config = bundle.get("config", None)

        # 帧缓冲（滑动窗口）
        self.buffer = deque(maxlen=window_size)
        self.visibility_buffer = deque(maxlen=window_size)  # 每帧平均可见性
        self.last_prob = 0.0
        self.last_prediction = 0
        self.predictions_history = deque(maxlen=10)  # 近期预测（防抖）
        self.frame_count = 0
        self.last_inference_time = 0.0

        # 统计
        self.total_inferences = 0
        self.total_inference_time = 0.0

        # 置信度衰减计数器（嵌赛项目: 连续判跌 + 置信度加权）
        self.fall_confidence = 0.0  # 0-1, 累积跌倒置信度
        self.confidence_decay = 0.80  # 正常帧衰减系数
        self.confidence_gain = 0.35   # 跌倒帧增益系数
        self.confidence_threshold = 0.65  # 累积置信度触发阈值

        print(f"[MLDetector] 模型加载完成")
        print(f"  类型: {type(self.model).__name__}")
        print(f"  窗口: {window_size} 帧, 步长: {stride}")
        print(f"  阈值: {threshold}")
        print(f"  决策: 置信度加权融合 (衰减={self.confidence_decay}, 触发={self.confidence_threshold})")
        if self.metrics.get("eval"):
            ev = self.metrics["eval"]
            print(f"  训练指标: F1={ev.get('f1', '?'):.4f} "
                  f"Recall={ev.get('recall', '?'):.4f} "
                  f"Precision={ev.get('precision', '?'):.4f}")

    def update(self, landmarks: np.ndarray) -> Tuple[float, int, bool]:
        """
        每帧调用：追加关键点 → 达到窗口大小时推理 → 置信度加权判定

        Args:
            landmarks: MediaPipe 33关键点 (33, 4) [x, y, z, visibility]

        Returns:
            (probability, label, is_fall)
                probability: 跌倒概率 0-1
                label: 0=正常, 1=疑似跌倒
                is_fall: 累积置信度 >= 触发阈值
        """
        self.frame_count += 1

        # 追加到缓冲区
        if landmarks.shape == (33, 4):
            buf_entry = landmarks[:, :3].copy()  # 只用 xyz
            avg_vis = float(np.mean(landmarks[:, 3]))  # 平均可见性
        elif landmarks.shape == (33, 3):
            buf_entry = landmarks.copy()
            avg_vis = 1.0
        else:
            raise ValueError(f"关键点形状错误: {landmarks.shape}, 需要 (33, 3) 或 (33, 4)")

        self.buffer.append(buf_entry)
        self.visibility_buffer.append(avg_vis)

        # 未满窗口，返回上次结果
        if len(self.buffer) < self.window_size:
            return self.last_prob, self.last_prediction, self.fall_confidence >= self.confidence_threshold

        # 每隔 stride 帧推理一次
        if self.frame_count % self.stride == 0:
            t0 = time.time()
            prob = self._infer()
            self.last_inference_time = time.time() - t0
            self.total_inferences += 1
            self.total_inference_time += self.last_inference_time

            # 置信度加权: 可见性低时降低概率
            avg_buffer_vis = np.mean(list(self.visibility_buffer)) if self.visibility_buffer else 1.0
            confidence_weight = min(1.0, 0.5 + 0.5 * avg_buffer_vis)  # 可见性 0→权重 0.5, 可见性 1→权重 1.0
            weighted_prob = prob * confidence_weight

            self.last_prob = prob  # 保存原始概率（用于调试）
            self.last_prediction = 1 if weighted_prob >= self.threshold else 0
            self.predictions_history.append(self.last_prediction)

            # 累积置信度: 判跌时增加, 正常时衰减（嵌赛项目: 连续置信度融合）
            if self.last_prediction == 1:
                self.fall_confidence = min(1.0, self.fall_confidence + weighted_prob * self.confidence_gain)
            else:
                self.fall_confidence = max(0.0, self.fall_confidence * self.confidence_decay)

            is_fall = self.fall_confidence >= self.confidence_threshold
        else:
            # 非推理帧: 置信度保持不变（只在推理帧更新）
            is_fall = self.fall_confidence >= self.confidence_threshold

        return self.last_prob, self.last_prediction, is_fall

    def _infer(self) -> float:
        """从当前窗口提取特征并推理"""
        window = np.array(list(self.buffer))  # (window_size, 33, 3)
        features = self._extract_window_features(window)

        # 标准化
        features_s = self.scaler.transform(features.reshape(1, -1))

        # 推理
        if hasattr(self.model, "predict_proba"):
            prob = self.model.predict_proba(features_s)[0, 1]
        else:
            prob = float(self.model.predict(features_s)[0])

        return float(prob)

    def _extract_window_features(self, lm_seq: np.ndarray) -> np.ndarray:
        """从关键点窗口提取特征（精简版，与训练时一致）"""
        T = len(lm_seq)
        x, y = lm_seq[:, :, 0], lm_seq[:, :, 1]

        # 索引常量
        NOSE = 0
        L_SHOULDER, R_SHOULDER = 11, 12
        L_HIP, R_HIP = 23, 24
        L_KNEE, R_KNEE = 25, 26
        L_ANKLE, R_ANKLE = 27, 28
        L_ELBOW, R_ELBOW = 13, 14
        L_WRIST, R_WRIST = 15, 16

        f = {}

        # 头部
        nose_y = y[:, NOSE]
        nose_vy = np.gradient(nose_y)
        f["head_y_min"] = float(np.min(nose_y))
        f["head_y_max"] = float(np.max(nose_y))
        f["head_y_range"] = float(np.ptp(nose_y))
        f["head_y_drop"] = float(nose_y[0] - nose_y[-1])
        f["head_vy_max"] = float(np.max(np.abs(nose_vy)))
        f["head_vy_mean"] = float(np.mean(np.abs(nose_vy)))

        # 躯干
        shoulder_mid_y = (y[:, L_SHOULDER] + y[:, R_SHOULDER]) / 2
        hip_mid_y = (y[:, L_HIP] + y[:, R_HIP]) / 2
        torso_center_y = (shoulder_mid_y + hip_mid_y) / 2
        shoulder_mid_x = (x[:, L_SHOULDER] + x[:, R_SHOULDER]) / 2
        hip_mid_x = (x[:, L_HIP] + x[:, R_HIP]) / 2

        f["torso_y_drop"] = float(torso_center_y[0] - torso_center_y[-1])
        f["torso_y_min"] = float(np.min(torso_center_y))
        f["torso_y_range"] = float(np.ptp(torso_center_y))

        dx = hip_mid_x - shoulder_mid_x
        dy = hip_mid_y - shoulder_mid_y + 1e-6
        torso_angles = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
        f["torso_angle_max"] = float(np.max(torso_angles))
        f["torso_angle_mean"] = float(np.mean(torso_angles))
        f["torso_angle_final"] = float(torso_angles[-1])
        f["torso_angle_change"] = float(np.ptp(torso_angles))

        # 运动中心
        all_y_mid = np.mean(y[:, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
                                   L_KNEE, R_KNEE]], axis=1)
        centroid_disp = np.sqrt(np.diff(all_y_mid, prepend=all_y_mid[0:1]) ** 2)
        f["centroid_disp_max"] = float(np.max(centroid_disp))
        f["centroid_disp_mean"] = float(np.mean(centroid_disp))
        f["centroid_disp_std"] = float(np.std(centroid_disp))
        f["centroid_total_disp"] = float(all_y_mid[-1] - all_y_mid[0])

        # 扩散范围
        active_y_range = np.ptp(y, axis=1)
        f["spread_max"] = float(np.max(active_y_range))
        f["spread_mean"] = float(np.mean(active_y_range))
        f["spread_std"] = float(np.std(active_y_range))

        # 速度
        centroid_vy = np.abs(np.gradient(all_y_mid))
        f["speed_max"] = float(np.max(centroid_vy))
        f["speed_mean"] = float(np.mean(centroid_vy))
        f["speed_std"] = float(np.std(centroid_vy))
        peak_idx = np.argmax(centroid_vy) if f["speed_max"] > 0 else 0
        f["speed_peak_position"] = float(peak_idx / max(T, 1))
        accel = np.abs(np.gradient(centroid_vy))
        f["accel_max"] = float(np.max(accel))
        f["accel_mean"] = float(np.mean(accel))

        # 静止恢复
        tail_n = max(1, int(T * 0.25))
        f["stillness_tail"] = float(np.mean(centroid_vy[-tail_n:]))
        f["speed_ratio_tail_head"] = float(
            np.mean(centroid_vy[-tail_n:]) / (np.mean(centroid_vy[:tail_n]) + 1e-6)
        )

        # 下肢
        knee_mid_y = (y[:, L_KNEE] + y[:, R_KNEE]) / 2
        ankle_mid_y = (y[:, L_ANKLE] + y[:, R_ANKLE]) / 2
        f["knee_y_drop"] = float(knee_mid_y[0] - knee_mid_y[-1])
        f["ankle_y_range"] = float(np.ptp(ankle_mid_y))
        f["hip_knee_ankle_spread_start"] = float(
            np.ptp([hip_mid_y[0], knee_mid_y[0], ankle_mid_y[0]])
        )
        f["hip_knee_ankle_spread_end"] = float(
            np.ptp([hip_mid_y[-1], knee_mid_y[-1], ankle_mid_y[-1]])
        )
        f["hip_knee_ankle_spread_min"] = float(
            min(np.min(hip_mid_y), np.min(knee_mid_y), np.min(ankle_mid_y))
        )

        # 上肢
        wrist_y_mean = np.mean(y[:, [L_WRIST, R_WRIST]], axis=1)
        elbow_y_mean = np.mean(y[:, [L_ELBOW, R_ELBOW]], axis=1)
        f["wrist_range"] = float(np.ptp(wrist_y_mean))
        f["elbow_wrist_dist"] = float(np.mean(np.abs(wrist_y_mean - elbow_y_mean)))

        # 头-躯干
        f["head_torso_offset"] = float(np.mean(np.abs(nose_y - torso_center_y)))

        # ---- 距离域特征 (雷达论文启发) ----
        centroid_vy_all = np.abs(np.gradient(all_y_mid))
        speed_weights = centroid_vy_all / (np.sum(centroid_vy_all) + 1e-6)
        f["motion_weighted_centroid"] = float(np.sum(speed_weights * all_y_mid))

        body_y_span = np.ptp(y, axis=1)
        start_n = min(5, T)
        end_n = min(5, T)
        f["active_range_start"] = float(np.mean(body_y_span[:start_n]))
        f["active_range_end"] = float(np.mean(body_y_span[-end_n:]))
        f["active_range_ratio"] = float(f["active_range_end"] / (f["active_range_start"] + 1e-6))

        total_displacement = float(np.abs(all_y_mid[-1] - all_y_mid[0]))
        f["speed_distance_ratio"] = float(f["speed_max"] / (total_displacement + 1e-6))
        f["body_compactness"] = float(np.mean(body_y_span) / (np.max(body_y_span) + 1e-6))

        # 按 feature_names 顺序排列
        features = np.array([f[name] for name in self.feature_names], dtype=np.float32)
        return features

    def reset(self):
        """重置状态（换人/重新开始）"""
        self.buffer.clear()
        self.visibility_buffer.clear()
        self.predictions_history.clear()
        self.last_prob = 0.0
        self.last_prediction = 0
        self.frame_count = 0
        self.fall_confidence = 0.0

    def get_stats(self) -> dict:
        avg_time = (self.total_inference_time / max(self.total_inferences, 1)) * 1000
        return {
            "total_inferences": self.total_inferences,
            "avg_ms": round(avg_time, 2),
            "buffer_fill": f"{len(self.buffer)}/{self.buffer.maxlen}",
            "last_prob": round(self.last_prob, 4),
            "fall_confidence": round(self.fall_confidence, 4),
            "weighted_decision": self.fall_confidence >= self.confidence_threshold,
        }


# ============================================================
# 独立测试
# ============================================================

def test_with_synthetic():
    """用模拟关键点测试推理器"""
    print("=== ML 推理器测试（模拟数据） ===\n")

    import sys, os as _os2
    _proj_root = _os2.path.dirname(_os2.path.dirname(_os2.path.abspath(__file__)))
    sys.path.insert(0, _os2.path.join(_proj_root, 'training'))
    from train_fall_classifier import SyntheticDataGenerator, FeatureExtractor

    # 先检查有没有训练好的模型
    model_path = "E:/老人跌倒/models/fall_classifier.pkl"
    if not os.path.exists(model_path):
        print(f"[ERROR] 请先训练模型: python train_fall_classifier.py --synthetic")
        return

    detector = MLFallDetector(model_path, window_size=30, stride=5, threshold=0.6)

    # 生成测试序列
    gen = SyntheticDataGenerator(n_frames_per_seq=60, n_sequences_per_class=10)
    test_cases = {
        "正向跌倒": gen.generate_fall_sequence(),
        "行走": gen.generate_walking_sequence(),
        "弯腰": gen.generate_bending_sequence(),
        "坐下": gen.generate_sitting_sequence(),
        "挥手": gen.generate_waving_sequence(),
        "蹲下": gen.generate_squatting_sequence(),
    }

    print(f"\n{'动作':<10} {'最高概率':>10} {'判定':>8} {'推理次数':>8} {'推理时间':>10}")
    print("-" * 55)

    for name, seq in test_cases.items():
        detector.reset()
        max_prob = 0.0
        fall_triggered = False

        for frame_lm in seq:
            # 加可见性伪列
            lm_4 = np.zeros((33, 4))
            lm_4[:, :3] = frame_lm
            lm_4[:, 3] = 1.0

            prob, _, is_fall = detector.update(lm_4)
            max_prob = max(max_prob, prob)
            if is_fall:
                fall_triggered = True

        stats = detector.get_stats()
        status = "✅" if (fall_triggered and "跌倒" in name) or (not fall_triggered and "跌倒" not in name) else "⚠️"

        print(f"  {status} {name:<8} {max_prob:>10.4f} {'FALL' if fall_triggered else 'OK':>8} "
              f"{stats['total_inferences']:>8} {stats['avg_ms']:>8.1f}ms")

    print("\n=== 测试完成 ===")


def main():
    parser = argparse.ArgumentParser(description="ML 跌倒推理模块")
    parser.add_argument("--model", "-m", type=str,
                        default="E:/老人跌倒/models/fall_classifier.pkl",
                        help="模型路径")
    parser.add_argument("--test", action="store_true",
                        help="用模拟数据测试推理器")
    parser.add_argument("--threshold", "-t", type=float, default=0.6,
                        help="判定阈值")
    args = parser.parse_args()

    if args.test:
        test_with_synthetic()
    else:
        # 快速验证模型加载
        detector = MLFallDetector(args.model, threshold=args.threshold)
        print("\n[OK] 模型加载成功，准备就绪！")
        print("  集成示例:")
        print("    from ml_fall_detector import MLFallDetector")
        print("    detector = MLFallDetector('E:/老人跌倒/models/fall_classifier.pkl')")
        print("    prob, label, is_fall = detector.update(landmarks_33x4)")


if __name__ == "__main__":
    main()
