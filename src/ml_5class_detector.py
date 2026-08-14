#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_5class_detector.py — 五分类行为识别推理模块
==============================================
加载训练好的 LightGBM 五分类模型，对实时关键点流做滑动窗口推理。

五分类: Fall(0) / SitDown(1) / StandUp(2) / Walking(3) / WakeUp(4)

与 e2e_fall_monitor.py 集成方式:
    detector = ML5ClassDetector("E:/老人跌倒/models/fall_classifier_5class.pkl")
    result = detector.update(landmarks_33x4)  # 每帧调用
    # result.class_name = "Walking" / "Fall" / ...
    # result.is_fall = True/False
    # result.probs = [0.01, 0.02, 0.05, 0.88, 0.04]

用法:
    python -m src.ml_5class_detector --test
"""

import argparse
import os
import pickle
import sys
import time
from typing import Optional, Tuple, List
from collections import deque, OrderedDict
from dataclasses import dataclass
import numpy as np

# ── 路径: 确保能 import training 模块的 FeatureExtractor ──
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_proj_root, 'training'))

# pickle 反序列化兼容: 模型可能存了 TrainConfig
try:
    from train_fall_classifier import TrainConfig, FeatureExtractor
except ImportError:
    TrainConfig = type('TrainConfig', (), {})
    FeatureExtractor = None
import __main__ as _main
_main.TrainConfig = TrainConfig

# 如果 FeatureExtractor 导入失败，回退到这里的内联版本
if FeatureExtractor is None:
    from train_fall_classifier import FeatureExtractor


# 类别 ID → 名称映射
CLASS_NAMES = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp"]
CLASS_LABELS_CN = {
    0: "摔倒",
    1: "坐下",
    2: "站起",
    3: "走路",
    4: "睡醒",
}


@dataclass
class DetectionResult:
    """单帧检测结果"""
    probs: np.ndarray          # (5,) 各类概率
    class_id: int              # 预测类别 0-4
    class_name: str            # "Fall" / "Walking" 等
    class_name_cn: str         # "摔倒" / "走路" 等
    is_fall: bool              # 是否判定为摔倒
    fall_prob: float           # 摔倒类别概率
    fall_confidence: float     # 累积置信度 (0-1)
    fall_triggered: bool       # 累积置信度 >= 触发阈值
    inference_done: bool       # 本帧是否执行了推理


class ML5ClassDetector:
    """基于 LightGBM 的五分类行为识别推理器"""

    def __init__(self, model_path: str,
                 window_size: int = 30,
                 stride: int = 5,
                 fall_threshold: float = 0.6):
        """
        Args:
            model_path: fall_classifier_5class.pkl 路径
            window_size: 滑动窗口帧数
            stride: 推理步长（每隔 N 帧推理一次）
            fall_threshold: 摔倒判定概率阈值
        """
        self.window_size = window_size
        self.stride = stride
        self.fall_threshold = fall_threshold

        # ── 加载模型 ──
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.classes = bundle.get("classes", CLASS_NAMES)
        self.n_classes = len(self.classes)
        self.feature_dim = bundle.get("feature_dim", 42)

        # 特征提取器（与训练时一致，42维）
        self.extractor = FeatureExtractor(window_size=window_size)

        # ── 缓冲区 ──
        self.buffer = deque(maxlen=window_size)
        self.visibility_buffer = deque(maxlen=window_size)

        # ── 状态 ──
        self.frame_count = 0
        self.last_result: Optional[DetectionResult] = None
        self.last_inference_time = 0.0

        # 累积置信度（针对摔倒类别）
        self._fall_confidence = 0.0
        self._conf_decay = 0.80
        self._conf_gain = 0.35
        self._conf_threshold = 0.65

        # 动作持久化：95% 窗口投票抑制闪烁
        self._prediction_window = deque(maxlen=100)
        self._stable_class = -1

        # 统计
        self.total_inferences = 0
        self.total_inference_time = 0.0

        # 运行历史（每 stride 记录一次）
        self.history: List[DetectionResult] = []

        print(f"[5ClassDetector] 五分类模型加载完成")
        print(f"  类型: {type(self.model).__name__}")
        print(f"  类别: {self.classes}")
        print(f"  特征维度: {self.feature_dim}")
        print(f"  窗口: {window_size} 帧, 步长: {stride}")
        print(f"  摔倒阈值: {fall_threshold}")

    # ──────────────────────────────────────────────
    # 主接口: 每帧调用
    # ──────────────────────────────────────────────

    def update(self, landmarks: np.ndarray) -> DetectionResult:
        """
        每帧调用：追加关键点 → 达到窗口大小时推理

        Args:
            landmarks: MediaPipe 33关键点 (33, 4) 或 (33, 3)

        Returns:
            DetectionResult 包含完整的五分类结果
        """
        self.frame_count += 1

        # ── 追加到缓冲区 ──
        if landmarks.shape == (33, 4):
            buf_entry = landmarks[:, :3].copy()
            avg_vis = float(np.mean(landmarks[:, 3]))
        elif landmarks.shape == (33, 3):
            buf_entry = landmarks.copy()
            avg_vis = 1.0
        else:
            raise ValueError(f"关键点形状错误: {landmarks.shape}")

        self.buffer.append(buf_entry)
        self.visibility_buffer.append(avg_vis)

        # ── 未满窗口，返回上次结果 ──
        if len(self.buffer) < self.window_size:
            if self.last_result is not None:
                return self._clone_result(self.last_result, inference_done=False)
            return self._empty_result()

        # ── 每隔 stride 帧推理一次 ──
        inference_done = False
        if self.frame_count % self.stride == 0:
            t0 = time.time()
            probs = self._infer()
            self.last_inference_time = time.time() - t0
            self.total_inferences += 1
            self.total_inference_time += self.last_inference_time
            inference_done = True

            # 可见性加权
            avg_buffer_vis = np.mean(list(self.visibility_buffer)) if self.visibility_buffer else 1.0
            weight = min(1.0, 0.5 + 0.5 * avg_buffer_vis)
            fall_prob = float(probs[0])  # class 0 = Fall

            # 累积置信度
            if fall_prob >= self.fall_threshold:
                self._fall_confidence = min(1.0, self._fall_confidence + fall_prob * weight * self._conf_gain)
            else:
                self._fall_confidence = max(0.0, self._fall_confidence * self._conf_decay)

            fall_triggered = self._fall_confidence >= self._conf_threshold

            # ── 动作持久化: 95% 窗口内多数投票 ──
            class_id = int(np.argmax(probs))
            self._prediction_window.append(class_id)
            self._stable_class = self._majority_vote(self._prediction_window, threshold=0.95)

            # 构建结果
            class_name = self.classes[self._stable_class] if self._stable_class >= 0 else "Unknown"
            class_name_cn = CLASS_LABELS_CN.get(self._stable_class, "未知")

            result = DetectionResult(
                probs=probs,
                class_id=self._stable_class,
                class_name=class_name,
                class_name_cn=class_name_cn,
                is_fall=(self._stable_class == 0 and fall_triggered),
                fall_prob=fall_prob,
                fall_confidence=self._fall_confidence,
                fall_triggered=fall_triggered,
                inference_done=True,
            )
            self.last_result = result
            self.history.append(result)

            # 保持历史在合理大小
            if len(self.history) > 500:
                self.history = self.history[-200:]

            return result

        # 非推理帧: 复用上次结果
        if self.last_result is not None:
            r = self._clone_result(self.last_result, inference_done=False)
            # 持续更新是否为摔倒（置信度可能因上一推理帧而改变）
            r.fall_triggered = self._fall_confidence >= self._conf_threshold
            return r
        return self._empty_result()

    # ──────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────

    def _infer(self) -> np.ndarray:
        """从当前窗口提取特征并推理，返回 (n_classes,) 概率向量"""
        window = np.array(list(self.buffer))
        features = self.extractor.extract_window(window)[0]  # (42,) 特征向量
        features_s = self.scaler.transform(features.reshape(1, -1))
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features_s)[0]
        else:
            # 回退: one-hot
            pred = int(self.model.predict(features_s)[0])
            probs = np.zeros(self.n_classes)
            probs[pred] = 1.0
        return probs.astype(np.float64)

    def _clone_result(self, r: DetectionResult, inference_done: bool) -> DetectionResult:
        return DetectionResult(
            probs=r.probs.copy(),
            class_id=r.class_id,
            class_name=r.class_name,
            class_name_cn=r.class_name_cn,
            is_fall=r.is_fall,
            fall_prob=r.fall_prob,
            fall_confidence=self._fall_confidence,
            fall_triggered=self._fall_confidence >= self._conf_threshold,
            inference_done=inference_done,
        )

    def _empty_result(self) -> DetectionResult:
        return DetectionResult(
            probs=np.zeros(self.n_classes),
            class_id=-1,
            class_name="Unknown",
            class_name_cn="未知",
            is_fall=False,
            fall_prob=0.0,
            fall_confidence=0.0,
            fall_triggered=False,
            inference_done=False,
        )

    @staticmethod
    def _majority_vote(window: deque, threshold: float = 0.95) -> int:
        """在最近 N 帧中选择出现频率超过阈值的主类别"""
        if len(window) < 10:
            if len(window) > 0:
                return window[-1]
            return -1
        counts = {}
        for c in window:
            counts[c] = counts.get(c, 0) + 1
        best = max(counts, key=counts.get)
        if counts[best] / len(window) >= threshold:
            return best
        return window[-1]  # 回退到最近一帧

    # ──────────────────────────────────────────────
    # 状态查询
    # ──────────────────────────────────────────────

    def reset(self):
        """重置所有状态"""
        self.buffer.clear()
        self.visibility_buffer.clear()
        self.last_result = None
        self.frame_count = 0
        self._fall_confidence = 0.0
        self._prediction_window.clear()
        self._stable_class = -1

    def get_stats(self) -> dict:
        avg_ms = (self.total_inference_time / max(self.total_inferences, 1)) * 1000
        return {
            "total_inferences": self.total_inferences,
            "avg_ms": round(avg_ms, 2),
            "buffer_fill": f"{len(self.buffer)}/{self.buffer.maxlen}",
            "stable_class": self.classes[self._stable_class] if self._stable_class >= 0 else "Unknown",
            "stable_class_cn": CLASS_LABELS_CN.get(self._stable_class, "未知"),
            "fall_prob": round(self.last_result.fall_prob if self.last_result else 0, 4),
            "fall_confidence": round(self._fall_confidence, 4),
            "fall_triggered": self._fall_confidence >= self._conf_threshold,
            "class_probs": [round(float(p), 4) for p in (self.last_result.probs if self.last_result else np.zeros(self.n_classes))],
        }


# ============================================================
# 独立测试
# ============================================================

def test_loading():
    """测试模型加载"""
    model_path = r"E:\老人跌倒\models\fall_classifier_5class.pkl"
    if not os.path.exists(model_path):
        print(f"[ERROR] 模型文件不存在: {model_path}")
        return False

    detector = ML5ClassDetector(model_path)
    print("\n[OK] 模型加载成功！")

    # 用模拟数据跑一轮
    print("\n模拟推理测试 (10个随机窗口)...")
    for i in range(10):
        # 随机站立姿态 + 噪声
        lm = np.random.normal(0.5, 0.05, (33, 4))
        lm[:, 3] = np.clip(np.random.normal(0.8, 0.1, 33), 0, 1)
        result = detector.update(lm)
        if result.inference_done:
            print(f"  推理 #{i+1}: {result.class_name_cn} "
                  f"probs={[round(float(p), 3) for p in result.probs]}")

    stats = detector.get_stats()
    print(f"\n统计: {stats}")
    return True


def main():
    parser = argparse.ArgumentParser(description="五分类行为识别推理模块")
    parser.add_argument("--model", "-m", type=str,
                        default=r"E:\老人跌倒\models\fall_classifier_5class.pkl",
                        help="五分类模型路径")
    parser.add_argument("--test", action="store_true", help="测试模型加载")
    args = parser.parse_args()

    if args.test:
        test_loading()
    else:
        detector = ML5ClassDetector(args.model)
        print("\n[OK] 五分类推理器就绪！")


if __name__ == "__main__":
    main()
