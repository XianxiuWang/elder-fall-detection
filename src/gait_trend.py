#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gait_trend.py — 步态趋势分析层（滑动窗口 + 风险评分）
=====================================================

v2.0 更新:
  - 所有参数可从 fall_config.GaitTrendConfig 注入
  - 行走状态判定阈值可配置
  - 支持参数热更新

用法:
    from .fall_config import GaitTrendConfig
    from .gait_trend import GaitTrendAnalyzer

    cfg = GaitTrendConfig()
    analyzer = GaitTrendAnalyzer(config=cfg)
"""

import math
import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING
from collections import deque
import numpy as np

if TYPE_CHECKING:
    from .fall_config import GaitTrendConfig


@dataclass
class GaitSample:
    """一帧的步态指标"""
    timestamp: float = 0.0
    step_length: float = 0.0
    walking_speed: float = 0.0
    sway_angle: float = 0.0
    balance_index: float = 0.0
    torso_angle: float = 0.0
    centroid_height: float = 0.0
    knee_angle_avg: float = 0.0
    is_walking: bool = False
    raw_features: Optional[np.ndarray] = None


@dataclass
class TrendReport:
    """一次趋势分析报告"""
    risk_score: float = 0.0
    baseline_deviation: float = 0.0
    trend_slope: float = 0.0
    instability: float = 0.0
    current_metrics: dict = field(default_factory=dict)
    baseline_metrics: dict = field(default_factory=dict)
    window_size: int = 0
    alert_level: str = "green"


# ============================================================
# 滑动窗口
# ============================================================

class GaitWindow:
    """滑动窗口管理器 — 维护最近 N 天的步态数据。"""

    def __init__(self, window_days: int = 7, samples_per_day: int = 100,
                 min_samples_for_analysis: int = 50):
        self.window_days = window_days
        self.samples_per_day = samples_per_day
        self.min_samples = min_samples_for_analysis
        self.max_samples = window_days * samples_per_day
        self._samples: deque[GaitSample] = deque(maxlen=self.max_samples)

    def add(self, sample: GaitSample) -> None:
        self._samples.append(sample)
        self._prune()

    def _prune(self) -> None:
        if not self._samples:
            return
        cutoff = time.time() - self.window_days * 24 * 3600
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def is_ready(self) -> bool:
        return len(self._samples) >= self.min_samples

    def get_all(self) -> List[GaitSample]:
        return list(self._samples)

    def get_walking_samples(self) -> List[GaitSample]:
        return [s for s in self._samples if s.is_walking]

    def get_recent(self, seconds: float) -> List[GaitSample]:
        cutoff = time.time() - seconds
        return [s for s in self._samples if s.timestamp >= cutoff]

    @property
    def size(self) -> int:
        return len(self._samples)

    def to_dict(self) -> dict:
        return {
            "window_days": self.window_days,
            "samples": [
                {"ts": s.timestamp, "step_length": s.step_length,
                 "walking_speed": s.walking_speed, "sway_angle": s.sway_angle,
                 "balance_index": s.balance_index, "torso_angle": s.torso_angle,
                 "centroid_height": s.centroid_height, "knee_angle_avg": s.knee_angle_avg,
                 "is_walking": s.is_walking}
                for s in self._samples
            ]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GaitWindow":
        win = cls(window_days=d["window_days"])
        for s in d["samples"]:
            win._samples.append(GaitSample(
                timestamp=s["ts"], step_length=s["step_length"],
                walking_speed=s["walking_speed"], sway_angle=s["sway_angle"],
                balance_index=s["balance_index"], torso_angle=s["torso_angle"],
                centroid_height=s["centroid_height"], knee_angle_avg=s["knee_angle_avg"],
                is_walking=s["is_walking"]))
        return win

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "GaitWindow":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ============================================================
# 单帧步态指标提取
# ============================================================

class GaitMetricExtractor:
    """从特征向量中计算步态指标"""

    def __init__(self, config: "GaitTrendConfig" = None):
        self.cfg = config

    def extract(self, features: np.ndarray, landmarks: Optional[np.ndarray] = None,
                prev_landmarks: Optional[np.ndarray] = None) -> GaitSample:
        sample = GaitSample(timestamp=time.time())
        if features is None or len(features) < 30:
            return sample

        feat = features
        sample.torso_angle = float(feat[3]) if len(feat) > 3 else 0.0
        sample.centroid_height = float(feat[1]) if len(feat) > 1 else 0.0

        if len(feat) > 73:
            sample.knee_angle_avg = (float(feat[72]) + float(feat[73])) / 2.0
        else:
            sample.knee_angle_avg = 0.0

        if len(feat) > 83:
            symmetry = float(feat[83])
            sample.balance_index = max(0.0, 1.0 - symmetry * 5)
        else:
            sample.balance_index = 0.5

        if landmarks is not None:
            sample.step_length = self._compute_step_length(landmarks)
        else:
            sample.step_length = 0.0

        if prev_landmarks is not None and landmarks is not None:
            sample.walking_speed = self._compute_walking_speed(landmarks, prev_landmarks)
        else:
            sample.walking_speed = 0.0

        if prev_landmarks is not None and landmarks is not None and feat is not None and len(feat) > 91:
            sample.sway_angle = float(feat[91])
        else:
            sample.sway_angle = sample.torso_angle

        sample.is_walking = self._is_walking_state(sample)
        return sample

    def _compute_step_length(self, landmarks: np.ndarray) -> float:
        left_ankle, right_ankle = landmarks[27], landmarks[28]
        step = abs(left_ankle[0] - right_ankle[0])
        left_hip, right_hip = landmarks[23], landmarks[24]
        hip_width = max(abs(right_hip[0] - left_hip[0]), 0.02)
        return step / hip_width

    def _compute_walking_speed(self, curr: np.ndarray, prev: np.ndarray) -> float:
        curr_centroid = (curr[23][:2] + curr[24][:2]) / 2.0
        prev_centroid = (prev[23][:2] + prev[24][:2]) / 2.0
        return float(np.linalg.norm(curr_centroid - prev_centroid))

    def _is_walking_state(self, sample: GaitSample) -> bool:
        if self.cfg is not None:
            c = self.cfg
            return (c.walking_knee_min < sample.knee_angle_avg < c.walking_knee_max
                    and sample.torso_angle < c.walking_torso_max
                    and c.walking_height_min < sample.centroid_height < c.walking_height_max
                    and sample.walking_speed > c.walking_speed_min)
        # 默认阈值
        return (130.0 < sample.knee_angle_avg < 185.0
                and sample.torso_angle < 25.0
                and 0.20 < sample.centroid_height < 0.50
                and sample.walking_speed > 0.001)


# ============================================================
# 趋势分析器
# ============================================================

class TrendAnalyzer:
    """步态趋势分析器"""

    def __init__(self, config: "GaitTrendConfig" = None,
                 baseline: Optional[dict] = None,
                 baseline_history_days: int = None):
        self.cfg = config
        if config is not None:
            self.baseline = baseline or config.default_baseline.copy()
            self.baseline_days = baseline_history_days or config.baseline_history_days
            self.weights = config.trending_weights.copy()
            self.metric_weights = config.metric_weights.copy()
        else:
            self.baseline = baseline or {
                "step_length": 0.15, "walking_speed": 0.05, "sway_angle": 5.0,
                "balance_index": 0.95, "torso_angle": 3.0,
                "centroid_height": 0.35, "knee_angle_avg": 170.0,
            }
            self.baseline_days = baseline_history_days or 3
            self.weights = {"baseline": 0.30, "trend": 0.40, "instability": 0.30}
            self.metric_weights = {
                "step_length": 0.25, "walking_speed": 0.25, "sway_angle": 0.20,
                "balance_index": 0.10, "torso_angle": 0.10,
                "centroid_height": 0.05, "knee_angle_avg": 0.05,
            }
        self._baseline_calculated = baseline is not None

    def update_config(self, config: "GaitTrendConfig") -> None:
        """运行时热更新参数"""
        self.cfg = config
        self.weights = config.trending_weights.copy()
        self.metric_weights = config.metric_weights.copy()

    def analyze(self, window: GaitWindow) -> TrendReport:
        samples = window.get_all()
        walking = window.get_walking_samples()
        if len(samples) < 10:
            return self._empty_report()

        if not self._baseline_calculated and len(samples) >= 20:
            self._compute_baseline(walking or samples)

        recent = [s for s in (walking or samples)
                  if s.timestamp > time.time() - 24 * 3600]
        current = self._aggregate_metrics(recent) if recent else self._aggregate_metrics(samples[-50:])

        baseline_dev = self._score_baseline_deviation(current)
        trend_slope = self._score_trend(walking or samples)
        instability = self._score_instability(walking or samples)

        raw_risk = (baseline_dev * self.weights["baseline"]
                    + trend_slope * self.weights["trend"]
                    + instability * self.weights["instability"])
        risk_score = min(100.0, max(0.0, raw_risk))

        alert = self._alert_level(risk_score)
        return TrendReport(
            risk_score=round(risk_score, 1),
            baseline_deviation=round(baseline_dev, 1),
            trend_slope=round(trend_slope, 1),
            instability=round(instability, 1),
            current_metrics=current,
            baseline_metrics=self.baseline.copy(),
            window_size=len(walking or samples),
            alert_level=alert,
        )

    def _score_baseline_deviation(self, current: dict) -> float:
        max_dev = self.cfg.max_deviation_ratio if self.cfg else 3.0
        total_dev, total_weight = 0.0, 0.0
        for metric, weight in self.metric_weights.items():
            cur = current.get(metric, 0.0)
            base = self.baseline.get(metric, 0.001)
            if base == 0:
                continue
            dev = abs(cur - base) / max(abs(base), 0.001)
            total_dev += weight * min(dev, max_dev)
            total_weight += weight
        return min(100.0, (total_dev / total_weight) * 100.0) if total_weight > 0 else 0.0

    def _score_trend(self, samples: List[GaitSample]) -> float:
        if len(samples) < 10:
            return 0.0

        n = len(samples)
        x = np.arange(n).astype(np.float32)
        x_mean, x_std = x.mean(), x.std() or 1.0

        y_step = np.array([s.step_length for s in samples], dtype=np.float32)
        y_speed = np.array([s.walking_speed for s in samples], dtype=np.float32)
        y_sway = np.array([s.sway_angle for s in samples], dtype=np.float32)

        slope_step = self._linear_slope(x, y_step, x_mean, x_std)
        slope_speed = self._linear_slope(x, y_speed, x_mean, x_std)
        slope_sway = self._linear_slope(x, y_sway, x_mean, x_std)

        if self.cfg is not None:
            sm, ss, sw = self.cfg.step_slope_mult, self.cfg.speed_slope_mult, self.cfg.sway_slope_mult
        else:
            sm, ss, sw = 100.0, 100.0, 50.0

        trend_score = (
            self.metric_weights["step_length"] * min(100.0, max(0.0, -slope_step * sm))
            + self.metric_weights["walking_speed"] * min(100.0, max(0.0, -slope_speed * ss))
            + self.metric_weights["sway_angle"] * min(100.0, max(0.0, slope_sway * sw))
        )
        total = (self.metric_weights["step_length"]
                 + self.metric_weights["walking_speed"]
                 + self.metric_weights["sway_angle"])
        return min(100.0, trend_score / total) if total > 0 else 0.0

    def _score_instability(self, samples: List[GaitSample]) -> float:
        if len(samples) < 10:
            return 0.0

        cv_offset = self.cfg.instability_cv_offset if self.cfg else 0.05
        cv_mult = self.cfg.instability_cv_mult if self.cfg else 200.0

        metrics = {
            "step_length": [s.step_length for s in samples],
            "walking_speed": [s.walking_speed for s in samples],
            "sway_angle": [s.sway_angle for s in samples],
            "balance_index": [s.balance_index for s in samples],
        }

        instability, total_weight = 0.0, 0.0
        for name, values in metrics.items():
            arr = np.array(values, dtype=np.float32)
            mean, std = arr.mean(), arr.std()
            if mean > 0.0001:
                cv = std / abs(mean)
                score = max(0.0, min(100.0, (cv - cv_offset) * cv_mult))
            else:
                score = 50.0
            w = self.metric_weights.get(name, 0.1)
            instability += w * score
            total_weight += w
        return min(100.0, instability / max(total_weight, 0.01))

    def _compute_baseline(self, samples: List[GaitSample]) -> None:
        if len(samples) < 5:
            return
        cutoff = time.time() - self.baseline_days * 24 * 3600
        early = [s for s in samples if s.timestamp < cutoff]
        if len(early) < 5:
            early = samples[:max(3, len(samples) // 2)]
        metrics = self._aggregate_metrics(early)
        for key in self.baseline:
            if key in metrics and metrics[key] != 0:
                self.baseline[key] = metrics[key]
        self._baseline_calculated = True

    @staticmethod
    def _aggregate_metrics(samples: List[GaitSample]) -> dict:
        if not samples:
            return {}
        result = {}
        keys = ["step_length", "walking_speed", "sway_angle",
                "balance_index", "torso_angle", "centroid_height", "knee_angle_avg"]
        for key in keys:
            values = [getattr(s, key, 0.0) for s in samples]
            arr = np.array(values, dtype=np.float32)
            result[key] = float(arr.mean())
            result[f"{key}_std"] = float(arr.std())
        return result

    @staticmethod
    def _linear_slope(x: np.ndarray, y: np.ndarray, x_mean: float, x_std: float) -> float:
        if x_std < 1e-6:
            return 0.0
        y_mean = y.mean()
        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()
        if denominator < 1e-6:
            return 0.0
        slope = float(numerator / denominator)
        if abs(y_mean) > 1e-6:
            slope /= abs(y_mean)
        return slope

    def _alert_level(self, score: float) -> str:
        levels = (self.cfg.alert_levels if self.cfg else
                  {(0, 25): "green", (25, 50): "yellow", (50, 75): "orange", (75, 101): "red"})
        for (lo, hi), level in sorted(levels.items()):
            if lo <= score < hi:
                return level
        return "red"

    def _empty_report(self) -> TrendReport:
        return TrendReport(
            risk_score=0.0, baseline_deviation=0.0,
            trend_slope=0.0, instability=0.0,
            current_metrics={}, baseline_metrics=self.baseline.copy(),
            window_size=0, alert_level="green")

    def save_state(self, path: str) -> None:
        state = {"baseline": self.baseline, "baseline_calculated": self._baseline_calculated,
                 "weights": self.weights}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_state(cls, path: str) -> "TrendAnalyzer":
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        analyzer = cls(baseline=state["baseline"])
        analyzer.weights = state.get("weights", analyzer.weights)
        analyzer._baseline_calculated = state["baseline_calculated"]
        return analyzer


# ============================================================
# 全家桶
# ============================================================

class GaitTrendAnalyzer:
    """
    步态趋势分析全家桶 — 一步到位。
    """

    def __init__(self, config: "GaitTrendConfig" = None, window_days: int = None,
                 samples_per_day: int = None, baseline_history_days: int = None,
                 data_dir: str = "gait_data"):
        if config is not None:
            self.cfg = config
            self.window = GaitWindow(config.window_days, config.samples_per_day,
                                     config.min_samples_for_analysis)
            self.extractor = GaitMetricExtractor(config=config)
            self.trend = TrendAnalyzer(config=config,
                                       baseline_history_days=config.baseline_history_days)
            self._analysis_interval = config.analysis_interval
        else:
            self.cfg = None
            wd = window_days or 7
            spd = samples_per_day or 100
            self.window = GaitWindow(wd, spd)
            self.extractor = GaitMetricExtractor()
            self.trend = TrendAnalyzer(baseline_history_days=baseline_history_days or 3)
            self._analysis_interval = 100

        self.data_dir = data_dir
        self._report: Optional[TrendReport] = None
        self._sample_count = 0

    def update(self,
               features: Optional[np.ndarray] = None,
               landmarks: Optional[np.ndarray] = None,
               prev_landmarks: Optional[np.ndarray] = None) -> Optional[TrendReport]:
        sample = self.extractor.extract(features, landmarks, prev_landmarks)
        self.window.add(sample)
        self._sample_count += 1

        if self._sample_count % self._analysis_interval == 0 and self.window.is_ready():
            self._report = self.trend.analyze(self.window)
        return self._report

    def force_analyze(self) -> TrendReport:
        self._report = self.trend.analyze(self.window)
        return self._report

    @property
    def alert_level(self) -> str:
        return self._report.alert_level if self._report else "unknown"

    @property
    def risk_score(self) -> float:
        return self._report.risk_score if self._report else 0.0

    @property
    def sample_count(self) -> int:
        return self.window.size

    def save(self, label: str = "") -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"{self.data_dir}/{label}_" if label else f"{self.data_dir}/"
        os.makedirs(self.data_dir, exist_ok=True)
        self.window.save(f"{prefix}window_{ts}.json")
        self.trend.save_state(f"{prefix}trend_{ts}.json")

    def load(self, window_path: str, trend_path: str) -> None:
        self.window = GaitWindow.load(window_path)
        self.trend = TrendAnalyzer.load_state(trend_path)


# ============================================================
# 快速测试
# ============================================================

def _test():
    print("=" * 60)
    print("步态趋势分析器 自测")
    print("=" * 60)

    from .fall_config import GaitTrendConfig
    cfg = GaitTrendConfig()
    analyzer = GaitTrendAnalyzer(config=cfg)
    np.random.seed(42)

    samples_to_gen = 150
    print(f"\n生成 {samples_to_gen} 个模拟样本（步态逐渐恶化）...\n")

    for i in range(samples_to_gen):
        t = time.time() - (samples_to_gen - i) * 600
        deterioration = i / samples_to_gen

        lm = np.random.rand(33, 4).astype(np.float32)
        lm[:, 1] = np.linspace(0.1, 0.95, 33) + np.random.randn(33) * 0.02
        lm[23, 0] = 0.44
        lm[24, 0] = 0.56
        lm[23:25, 1] = 0.55

        prev_lm = lm.copy()
        prev_lm[23:25, 0] -= 0.002 * (1 - deterioration) + np.random.randn() * 0.001

        feat = np.zeros(100, dtype=np.float32)
        feat[1] = 0.35 + deterioration * 0.1
        feat[3] = 3.0 + deterioration * 15.0
        feat[72] = 170.0 - deterioration * 30.0
        feat[73] = 168.0 - deterioration * 30.0
        feat[83] = 0.05 + deterioration * 0.15
        feat[91] = 1.0 + deterioration * 8.0

        sample = analyzer.extractor.extract(feat, lm, prev_lm)
        sample.timestamp = t
        sample.is_walking = True
        sample.step_length = 0.15 - deterioration * 0.08
        sample.walking_speed = 0.05 - deterioration * 0.03
        sample.balance_index = 0.95 - deterioration * 0.20
        analyzer.window.add(sample)

    print(f"窗口样本数: {analyzer.window.size}")
    report = analyzer.trend.analyze(analyzer.window)

    print(f"\n{'─' * 40}")
    print(f"  风险分数:     {report.risk_score:.1f}/100")
    print(f"  偏离基线:     {report.baseline_deviation:.1f}")
    print(f"  趋势恶化:     {report.trend_slope:.1f}")
    print(f"  不稳定性:     {report.instability:.1f}")
    print(f"  窗口样本数:   {report.window_size}")
    print(f"  告警等级:     {report.alert_level.upper()}")
    print(f"{'─' * 40}")
    print(f"\n  基线 vs 当前:")
    for key in ["step_length", "walking_speed", "sway_angle", "torso_angle"]:
        base = report.baseline_metrics.get(key, 0)
        cur = report.current_metrics.get(key, 0)
        direction = ("↓恶化" if ((key in ("step_length", "walking_speed") and cur < base) or cur > base)
                     else "→")
        print(f"    {key:20s} | 基线:{base:7.3f} → 当前:{cur:7.3f} {direction}")


if __name__ == "__main__":
    _test()
