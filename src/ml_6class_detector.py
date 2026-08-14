#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ml_6class_detector.py — 六分类行为识别推理模块（含 Standing/Idle）
================================================================
加载训练好的 LightGBM 六分类模型，对实时关键点流做滑动窗口推理。

六分类: Fall(0) / SitDown(1) / StandUp(2) / Walking(3) / WakeUp(4) / Standing(5)

与 e2e_fall_monitor.py 集成方式:
    detector = ML6ClassDetector("E:/老人跌倒/models/fall_classifier_6class.pkl")
    result = detector.update(landmarks_33x4)  # 每帧调用
    # result.class_name = "Standing" / "Walking" / "Fall" / ...
    # result.is_fall = True/False
    # result.is_standing = True/False  ← 新增
"""

import argparse
import os
import pickle
import sys
import time
from typing import Optional, List
from collections import deque
from dataclasses import dataclass
import numpy as np

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_proj_root, 'training'))

try:
    from train_fall_classifier import TrainConfig, FeatureExtractor
except ImportError:
    TrainConfig = type('TrainConfig', (), {})
    FeatureExtractor = None

import __main__ as _main
_main.TrainConfig = TrainConfig

if FeatureExtractor is None:
    from train_fall_classifier import FeatureExtractor


CLASS_NAMES = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"]
CLASS_LABELS_CN = {
    0: "摔倒",
    1: "坐下",
    2: "站起",
    3: "走路",
    4: "睡醒",
    5: "站立",
}

FALL_CLASS_ID = 0
STANDING_CLASS_ID = 5


@dataclass
class DetectionResult:
    """单帧检测结果"""
    probs: np.ndarray          # (6,) 各类概率
    class_id: int              # 预测类别 0-5
    class_name: str            # "Fall" / "Standing" 等
    class_name_cn: str         # "摔倒" / "站立" 等
    is_fall: bool              # 是否判定为摔倒
    is_standing: bool          # 是否判定为站立
    fall_prob: float           # 摔倒类别概率
    standing_prob: float       # 站立类别概率
    fall_confidence: float     # 累积置信度 (0-1)
    fall_triggered: bool       # 累积置信度 >= 触发阈值
    inference_done: bool       # 本帧是否执行了推理
    inference_count: int = 0   # v6.1: 总推理次数 (用于预热判断)


class ML6ClassDetector:
    """基于 LightGBM 的六分类行为识别推理器（含 Standing/Idle）"""

    def __init__(self, model_path: str,
                 window_size: int = 30,
                 stride: int = 5,
                 fall_threshold: float = 0.6):
        self.window_size = window_size
        self.stride = stride
        self.fall_threshold = fall_threshold

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)

        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.classes = bundle.get("classes", CLASS_NAMES)
        self.n_classes = len(self.classes)
        self.feature_dim = bundle.get("feature_dim", 42)

        self.extractor = FeatureExtractor(window_size=window_size)
        self.buffer = deque(maxlen=window_size)
        self.visibility_buffer = deque(maxlen=window_size)

        self.frame_count = 0
        self.last_result: Optional[DetectionResult] = None
        self.last_inference_time = 0.0

        self._fall_confidence = 0.0
        self._conf_decay = 0.80
        self._conf_gain = 0.35
        self._conf_threshold = 0.65

        self._prediction_window = deque(maxlen=100)
        self._stable_class = -1

        self.total_inferences = 0
        self.total_inference_time = 0.0
        self.history: List[DetectionResult] = []

        # v6.1: 静默加载 (批量创建时不刷屏)
        # print(f"[6ClassDetector] 六分类模型加载完成")
        print(f"  类型: {type(self.model).__name__}")
        print(f"  类别: {self.classes}")
        print(f"  特征维度: {self.feature_dim}")
        print(f"  窗口: {window_size} 帧, 步长: {stride}")

    def update(self, landmarks: np.ndarray) -> DetectionResult:
        self.frame_count += 1

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

        if len(self.buffer) < self.window_size:
            if self.last_result is not None:
                return self._clone_result(self.last_result, inference_done=False)
            return self._empty_result()

        inference_done = False
        if self.frame_count % self.stride == 0:
            t0 = time.time()
            probs = self._infer()
            self.last_inference_time = time.time() - t0
            self.total_inferences += 1
            self.total_inference_time += self.last_inference_time
            inference_done = True

            avg_buffer_vis = np.mean(list(self.visibility_buffer)) if self.visibility_buffer else 1.0
            weight = min(1.0, 0.5 + 0.5 * avg_buffer_vis)
            fall_prob = float(probs[FALL_CLASS_ID])
            standing_prob = float(probs[STANDING_CLASS_ID])

            if fall_prob >= self.fall_threshold:
                self._fall_confidence = min(1.0, self._fall_confidence + fall_prob * weight * self._conf_gain)
            else:
                self._fall_confidence = max(0.0, self._fall_confidence * self._conf_decay)

            fall_triggered = self._fall_confidence >= self._conf_threshold

            class_id = int(np.argmax(probs))
            self._prediction_window.append(class_id)
            self._stable_class = self._majority_vote(self._prediction_window, threshold=0.95)

            class_name = self.classes[self._stable_class] if self._stable_class >= 0 else "Unknown"
            class_name_cn = CLASS_LABELS_CN.get(self._stable_class, "未知")

            result = DetectionResult(
                probs=probs,
                class_id=self._stable_class,
                class_name=class_name,
                class_name_cn=class_name_cn,
                is_fall=(self._stable_class == FALL_CLASS_ID and fall_triggered),
                is_standing=(self._stable_class == STANDING_CLASS_ID),
                fall_prob=fall_prob,
                standing_prob=standing_prob,
                fall_confidence=self._fall_confidence,
                fall_triggered=fall_triggered,
                inference_done=True,
                inference_count=self.total_inferences,
            )
            self.last_result = result
            self.history.append(result)

            if len(self.history) > 500:
                self.history = self.history[-200:]

            return result

        if self.last_result is not None:
            r = self._clone_result(self.last_result, inference_done=False)
            r.fall_triggered = self._fall_confidence >= self._conf_threshold
            return r
        return self._empty_result()

    def _infer(self) -> np.ndarray:
        window = np.array(list(self.buffer))
        features = self.extractor.extract_window(window)[0]
        features_s = self.scaler.transform(features.reshape(1, -1))
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features_s)[0]
        else:
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
            is_standing=r.is_standing,
            fall_prob=r.fall_prob,
            standing_prob=r.standing_prob,
            fall_confidence=self._fall_confidence,
            fall_triggered=self._fall_confidence >= self._conf_threshold,
            inference_done=inference_done,
            inference_count=self.total_inferences,
        )

    def _empty_result(self) -> DetectionResult:
        return DetectionResult(
            probs=np.zeros(self.n_classes),
            class_id=-1,
            class_name="Unknown",
            class_name_cn="未知",
            is_fall=False,
            is_standing=False,
            fall_prob=0.0,
            standing_prob=0.0,
            fall_confidence=0.0,
            fall_triggered=False,
            inference_done=False,
        )

    @staticmethod
    def _majority_vote(window: deque, threshold: float = 0.95) -> int:
        if len(window) < 10:
            return window[-1] if len(window) > 0 else -1
        counts = {}
        for c in window:
            counts[c] = counts.get(c, 0) + 1
        best = max(counts, key=counts.get)
        return best if counts[best] / len(window) >= threshold else window[-1]

    def reset(self):
        self.buffer.clear()
        self.visibility_buffer.clear()
        self.last_result = None
        self.frame_count = 0
        self._fall_confidence = 0.0
        self._prediction_window.clear()
        self._stable_class = -1

    def get_stats(self) -> dict:
        avg_ms = (self.total_inference_time / max(self.total_inferences, 1)) * 1000
        probs = self.last_result.probs if self.last_result else np.zeros(self.n_classes)
        return {
            "total_inferences": self.total_inferences,
            "avg_ms": round(avg_ms, 2),
            "buffer_fill": f"{len(self.buffer)}/{self.buffer.maxlen}",
            "stable_class": self.classes[self._stable_class] if self._stable_class >= 0 else "Unknown",
            "stable_class_cn": CLASS_LABELS_CN.get(self._stable_class, "未知"),
            "is_standing": self.last_result.is_standing if self.last_result else False,
            "fall_prob": round(self.last_result.fall_prob if self.last_result else 0, 4),
            "standing_prob": round(self.last_result.standing_prob if self.last_result else 0, 4),
            "fall_confidence": round(self._fall_confidence, 4),
            "fall_triggered": self._fall_confidence >= self._conf_threshold,
            "class_probs": [round(float(p), 4) for p in probs],
        }


def test_loading():
    """测试模型加载"""
    model_path = r"E:\老人跌倒\models\fall_classifier_6class.pkl"
    if not os.path.exists(model_path):
        print(f"[ERROR] 模型文件不存在: {model_path}")
        return False

    detector = ML6ClassDetector(model_path)
    print("\n[OK] 六分类模型加载成功！")

    print("\n模拟推理测试 (20个随机窗口 → 预期被识别为 Standing)...")
    for i in range(20):
        lm = np.random.normal(0.5, 0.03, (33, 4))
        lm[:, 3] = np.clip(np.random.normal(0.8, 0.1, 33), 0, 1)
        result = detector.update(lm)
        if result.inference_done:
            print(f"  推理 #{i+1}: {result.class_name_cn} "
                  f"is_standing={result.is_standing} "
                  f"probs={[round(float(p), 3) for p in result.probs]}")

    stats = detector.get_stats()
    print(f"\n统计: class={stats['stable_class_cn']}, "
          f"is_standing={stats['is_standing']}, "
          f"probs={stats['class_probs']}")
    return True


def main():
    parser = argparse.ArgumentParser(description="六分类行为识别推理模块")
    parser.add_argument("--model", "-m", type=str,
                        default=r"E:\老人跌倒\models\fall_classifier_6class.pkl",
                        help="六分类模型路径")
    parser.add_argument("--test", action="store_true", help="测试模型加载")
    args = parser.parse_args()

    if args.test:
        test_loading()
    else:
        detector = ML6ClassDetector(args.model)
        print("\n[OK] 六分类推理器就绪！")


if __name__ == "__main__":
    main()
