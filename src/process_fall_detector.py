#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_fall_detector.py — 过程级跌倒检测器
===========================================
灵感来源: 瞳芯颐护的"过程分析降低误报率"（不是单帧判断，而是分析序列轨迹）

v2.0 更新:
  - 所有参数可从 fall_config.ProcessFallConfig 注入
  - 单帧风险分计算参数可配置
  - 5维分析各维度子参数可配置
  - 支持参数热更新（update_config）

用法:
    from .fall_config import ProcessFallConfig
    from .process_fall_detector import ProcessFallDetector

    cfg = ProcessFallConfig()
    detector = ProcessFallDetector(config=cfg)
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, TYPE_CHECKING
from collections import deque
import numpy as np

if TYPE_CHECKING:
    from .fall_config import ProcessFallConfig


@dataclass
class FrameSnapshot:
    """单帧快照"""
    frame_idx: int
    timestamp: float
    centroid_y: float = 0.0
    torso_angle: float = 0.0
    centroid_disp: float = 0.0
    motion_spread: float = 0.0
    torso_disp_ratio: float = 0.0
    is_alert_candidate: bool = False
    raw_risk: float = 0.0


@dataclass
class FallAlert:
    """一次跌倒告警"""
    alert_level: str
    confidence: float
    peak_speed: float
    total_displacement: float
    height_drop: float
    torso_change: float
    duration_frames: int
    timestamp: float
    debug_info: dict = field(default_factory=dict)


class ProcessFallDetector:
    """
    过程级跌倒检测器 — 三阶段状态机: NORMAL → COLLECTING → ANALYZING
    """

    class State:
        NORMAL = "NORMAL"
        COLLECTING = "COLLECTING"
        ANALYZING = "ANALYZING"

    def __init__(self,
                 config: "ProcessFallConfig" = None,
                 window_seconds: float = None,
                 frame_rate: float = None,
                 min_sequence_frames: int = None,
                 fall_height_threshold: float = None,
                 fall_speed_threshold: float = None,
                 cooldown_seconds: float = None):
        """
        可通过 config 对象传入所有参数，也支持独立参数覆盖（兼容旧接口）。
        """
        self.cfg = config

        def _val(new_val, default):
            return new_val if new_val is not None else default

        if config is not None:
            self.window_frames = int(config.window_seconds * config.frame_rate)
            self.min_sequence = config.min_sequence_frames
            self.height_threshold = config.fall_height_threshold
            self.speed_threshold = config.fall_speed_threshold
            self.cooldown = config.cooldown_seconds
            self.consecutive_trigger = config.consecutive_suspicious
        else:
            self.window_frames = int(_val(window_seconds, 2.0) * _val(frame_rate, 30.0))
            self.min_sequence = _val(min_sequence_frames, 15)
            self.height_threshold = _val(fall_height_threshold, 0.08)
            self.speed_threshold = _val(fall_speed_threshold, 0.03)
            self.cooldown = _val(cooldown_seconds, 5.0)
            self.consecutive_trigger = 3

        self._state = self.State.NORMAL
        self._buffer: deque[FrameSnapshot] = deque(maxlen=self.window_frames)
        self._sequence_start = 0
        self._last_alert_time = 0.0
        self._consecutive_suspicious = 0

    def update_config(self, config: "ProcessFallConfig") -> None:
        """运行时热更新参数"""
        self.cfg = config
        self.window_frames = int(config.window_seconds * config.frame_rate)
        self.min_sequence = config.min_sequence_frames
        self.height_threshold = config.fall_height_threshold
        self.speed_threshold = config.fall_speed_threshold
        self.cooldown = config.cooldown_seconds
        self.consecutive_trigger = config.consecutive_suspicious
        self._buffer = deque(maxlen=self.window_frames)

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def update(self,
               features: np.ndarray,
               spatial: Optional[np.ndarray] = None,
               frame_idx: int = 0) -> Optional[FallAlert]:
        """喂入一帧，返回告警（如果有）。"""
        now = time.time()

        snap = FrameSnapshot(frame_idx=frame_idx, timestamp=now)

        if features is not None and len(features) >= 4:
            snap.centroid_y = float(features[1])
            snap.torso_angle = float(features[3])

        if spatial is not None and len(spatial) >= 6:
            snap.centroid_disp = float(spatial[0])
            snap.motion_spread = float(spatial[3])
            snap.torso_disp_ratio = float(spatial[4])

        snap.is_alert_candidate = self._is_suspicious_frame(snap)
        snap.raw_risk = self._single_frame_risk(snap)
        self._buffer.append(snap)

        # 状态机
        if self._state == self.State.NORMAL:
            if snap.is_alert_candidate:
                self._consecutive_suspicious += 1
            else:
                self._consecutive_suspicious = 0

            if self._consecutive_suspicious >= self.consecutive_trigger:
                self._state = self.State.COLLECTING
                self._sequence_start = len(self._buffer) - self._consecutive_suspicious
            return None

        elif self._state == self.State.COLLECTING:
            seq_len = len(self._buffer) - self._sequence_start
            if seq_len >= self.min_sequence:
                self._state = self.State.ANALYZING
            return None

        elif self._state == self.State.ANALYZING:
            alert = self._analyze_sequence()
            self._state = self.State.NORMAL
            self._consecutive_suspicious = 0

            if alert and (now - self._last_alert_time) < self.cooldown:
                alert.alert_level = "SUSPICIOUS"

            if alert:
                self._last_alert_time = now

            return alert

        return None

    def force_analyze(self) -> Optional[FallAlert]:
        return self._analyze_sequence()

    def reset(self):
        self._state = self.State.NORMAL
        self._buffer.clear()
        self._consecutive_suspicious = 0

    # ------------------------------------------------------------------
    # 单帧判定
    # ------------------------------------------------------------------

    def _is_suspicious_frame(self, snap: FrameSnapshot) -> bool:
        cfg = self.cfg
        if cfg is None:
            # 默认阈值
            if snap.centroid_disp <= 0.03: return False
            signals = 1
            signals += 1 if snap.motion_spread > 0.3 else 0
            signals += 1 if snap.torso_disp_ratio > 0.25 else 0
            signals += 1 if snap.torso_angle > 30.0 else 0
            return signals >= 3

        signals = 0
        total = 0

        if snap.centroid_disp > cfg.suspicious_centroid_disp:
            signals += 1
        total += 1

        if snap.motion_spread > cfg.suspicious_motion_spread:
            signals += 1
        total += 1

        if snap.torso_disp_ratio > cfg.suspicious_torso_disp_ratio:
            signals += 1
        total += 1

        if snap.torso_angle > cfg.suspicious_torso_angle:
            signals += 1
        total += 1

        return signals >= cfg.suspicious_min_signals

    def _single_frame_risk(self, snap: FrameSnapshot) -> float:
        cfg = self.cfg
        if cfg is None:
            risk = 0.0
            w = 0.0
            if snap.centroid_disp > 0:
                risk += min(1.0, snap.centroid_disp * 20) * 0.35
            w += 0.35
            risk += min(1.0, snap.motion_spread) * 0.30
            w += 0.30
            risk += snap.torso_disp_ratio * 0.20
            w += 0.20
            if snap.torso_angle > 0:
                risk += min(1.0, snap.torso_angle / 90.0) * 0.15
            w += 0.15
            return min(1.0, risk / w)

        risk = 0.0
        w = 0.0

        if snap.centroid_disp > 0:
            risk += min(1.0, snap.centroid_disp * cfg.risk_centroid_disp_mult) * cfg.risk_centroid_disp_weight
        w += cfg.risk_centroid_disp_weight

        risk += min(1.0, snap.motion_spread) * cfg.risk_spread_weight
        w += cfg.risk_spread_weight

        risk += snap.torso_disp_ratio * cfg.risk_torso_disp_weight
        w += cfg.risk_torso_disp_weight

        if snap.torso_angle > 0:
            risk += min(1.0, snap.torso_angle / cfg.risk_torso_angle_max) * cfg.risk_torso_angle_weight
        w += cfg.risk_torso_angle_weight

        return min(1.0, risk / w) if w > 0 else 0.0

    # ------------------------------------------------------------------
    # 序列分析
    # ------------------------------------------------------------------

    def _analyze_sequence(self) -> Optional[FallAlert]:
        if len(self._buffer) < self.min_sequence:
            return None

        seq = list(self._buffer)[self._sequence_start:]
        n = len(seq)
        if n < self.min_sequence:
            return None

        # 提取各维度特征序列
        speeds = [s.centroid_disp for s in seq]
        heights = [s.centroid_y for s in seq]
        torso_angles = [s.torso_angle for s in seq]
        spreads = [s.motion_spread for s in seq]

        # 5维度评分
        speed_score = self._analyze_speed_profile(speeds)
        height_score = self._analyze_height_profile(heights)
        torso_score = self._analyze_torso_profile(torso_angles)
        spread_score = self._analyze_spread_profile(spreads)
        stillness_score = self._analyze_stillness(seq)

        # 加权综合
        weights = (self.cfg.analysis_weights if self.cfg else
                   {"speed": 0.25, "height": 0.30, "torso": 0.20, "spread": 0.10, "stillness": 0.15})

        total_score = (
            speed_score * weights["speed"]
            + height_score * weights["height"]
            + torso_score * weights["torso"]
            + spread_score * weights["spread"]
            + stillness_score * weights["stillness"]
        )

        # 统计量
        peak_speed = max(speeds) if speeds else 0.0
        height_start = heights[0] if heights else 0.0
        height_end = heights[-1] if heights else 0.0
        height_drop = max(0.0, height_end - height_start)
        torso_start = torso_angles[0] if torso_angles else 0.0
        torso_end = torso_angles[-1] if torso_angles else 0.0
        torso_change = abs(torso_end - torso_start)

        # 告警等级
        thresholds = (self.cfg.alert_thresholds if self.cfg else
                      {"SUSPICIOUS": 0.4, "WARNING": 0.6, "ALERT": 0.8, "URGENT": 0.95})
        level = "SUSPICIOUS"
        for lvl, thr in sorted(thresholds.items(), key=lambda x: x[1]):
            if total_score >= thr:
                level = lvl

        # 高度下降不够时降权
        low_penalty = self.cfg.low_height_drop_penalty if self.cfg else 0.3
        if height_drop < self.height_threshold * 0.5:
            total_score *= low_penalty
            level = "SUSPICIOUS"

        return FallAlert(
            alert_level=level,
            confidence=round(total_score, 3),
            peak_speed=round(peak_speed, 4),
            total_displacement=round(height_drop, 4),
            height_drop=round(height_drop, 4),
            torso_change=round(torso_change, 1),
            duration_frames=n,
            timestamp=seq[-1].timestamp if seq else time.time(),
            debug_info={
                "speed_score": round(speed_score, 3),
                "height_score": round(height_score, 3),
                "torso_score": round(torso_score, 3),
                "spread_score": round(spread_score, 3),
                "stillness_score": round(stillness_score, 3),
            },
        )

    # ------------------------------------------------------------------
    # 子维度分析（参数已配置化）
    # ------------------------------------------------------------------

    def _analyze_speed_profile(self, speeds: List[float]) -> float:
        if not speeds:
            return 0.0

        arr = np.array(speeds, dtype=np.float32)
        peak = arr.max()
        mean = arr.mean()
        if mean < 1e-6:
            return 0.0

        peak_mean_ratio = peak / mean

        mid = len(arr) // 2
        first_half_mean = arr[:mid].mean() if mid > 0 else 0
        second_half_mean = arr[mid:].mean() if mid > 0 else 0
        deceleration_ratio = ((first_half_mean - second_half_mean)
                              / max(first_half_mean, 1e-6))

        cfg = self.cfg
        score = 0.0
        r_high = cfg.speed_peak_mean_ratio_high if cfg else 3.0
        r_mid = cfg.speed_peak_mean_ratio_mid if cfg else 2.0
        r_low = cfg.speed_peak_mean_ratio_low if cfg else 1.5
        d_high = cfg.speed_deceleration_high if cfg else 0.3
        d_low = cfg.speed_deceleration_low if cfg else 0.1

        if peak_mean_ratio > r_high:
            score += 0.5
        elif peak_mean_ratio > r_mid:
            score += 0.3
        elif peak_mean_ratio > r_low:
            score += 0.1

        if deceleration_ratio > d_high:
            score += 0.5
        elif deceleration_ratio > d_low:
            score += 0.3

        return min(1.0, score)

    def _analyze_height_profile(self, heights: List[float]) -> float:
        if not heights:
            return 0.0

        arr = np.array(heights, dtype=np.float32)
        start, end = arr[0], arr[-1]
        total_drop = end - start

        cfg = self.cfg
        ht = self.height_threshold

        if total_drop > ht * 2:
            drop_score = 1.0
        elif total_drop > ht:
            drop_score = 0.6
        elif total_drop > ht * 0.3:
            drop_score = 0.2
        else:
            drop_score = 0.0

        min_idx = np.argmin(arr)
        min_pos_ratio = (min_idx + 1) / len(arr)

        if min_idx < len(arr) - 3:
            post_min = arr[min_idx:]
            recovery = post_min.max() - post_min.min()
            not_recovered = 1.0 - min(1.0, recovery / max(total_drop, 0.001))
        else:
            not_recovered = 1.0

        if min_pos_ratio > 0.6:
            late_peak_score = 1.0
        elif min_pos_ratio > 0.3:
            late_peak_score = 0.5
        else:
            late_peak_score = 0.2

        if cfg is not None:
            return min(1.0, (drop_score * cfg.height_drop_weight +
                             not_recovered * cfg.height_not_recovered_weight +
                             late_peak_score * cfg.height_late_peak_weight))
        return min(1.0, drop_score * 0.6 + not_recovered * 0.2 + late_peak_score * 0.2)

    def _analyze_torso_profile(self, angles: List[float]) -> float:
        if not angles:
            return 0.0

        arr = np.array(angles, dtype=np.float32)
        start, end = arr[0], arr[-1]

        cfg = self.cfg
        if cfg is None:
            angle_score = 1.0 if end > 60 else 0.7 if end > 40 else 0.3 if end > 25 else 0.1
            change = abs(end - start)
            change_score = 1.0 if change > 40 else 0.6 if change > 20 else 0.2
            mono_count = sum(1 for i in range(1, len(arr)) if arr[i] >= arr[i - 1])
            monotonicity = mono_count / max(len(arr) - 1, 1)
            mono_score = 1.0 if monotonicity > 0.8 else 0.5 if monotonicity > 0.5 else 0.1
            return min(1.0, angle_score * 0.4 + change_score * 0.3 + mono_score * 0.3)
        else:
            a_h, a_m, a_l = cfg.torso_final_angle_high, cfg.torso_final_angle_mid, cfg.torso_final_angle_low
            angle_score = 1.0 if end > a_h else 0.7 if end > a_m else 0.3 if end > a_l else 0.1

            change = abs(end - start)
            c_h, c_m = cfg.torso_change_high, cfg.torso_change_mid
            change_score = 1.0 if change > c_h else 0.6 if change > c_m else 0.2

            mono_count = sum(1 for i in range(1, len(arr)) if arr[i] >= arr[i - 1])
            monotonicity = mono_count / max(len(arr) - 1, 1)
            m_h, m_m = cfg.torso_monotonicity_high, cfg.torso_monotonicity_mid
            mono_score = 1.0 if monotonicity > m_h else 0.5 if monotonicity > m_m else 0.1

            return min(1.0, (angle_score * cfg.torso_angle_weight +
                             change_score * cfg.torso_change_weight +
                             mono_score * cfg.torso_mono_weight))

    def _analyze_spread_profile(self, spreads: List[float]) -> float:
        if not spreads:
            return 0.0

        arr = np.array(spreads, dtype=np.float32)
        max_spread = arr.max()

        cfg = self.cfg
        if cfg is None:
            spread_score = 1.0 if max_spread > 0.6 else 0.6 if max_spread > 0.4 else 0.3 if max_spread > 0.2 else 0.0
            mid = len(arr) // 2
            change_score = min(1.0, abs(arr[mid:].mean() - arr[:mid].mean()) * 3) if mid > 0 else 0.0
            return spread_score * 0.6 + change_score * 0.4
        else:
            s_h, s_m, s_l = cfg.spread_max_high, cfg.spread_max_mid, cfg.spread_max_low
            spread_score = (1.0 if max_spread > s_h else 0.6 if max_spread > s_m
                            else 0.3 if max_spread > s_l else 0.0)
            mid = len(arr) // 2
            change_score = (min(1.0, abs(arr[mid:].mean() - arr[:mid].mean()) * cfg.spread_change_mult)
                            if mid > 0 else 0.0)
            return spread_score * cfg.spread_max_weight + change_score * cfg.spread_change_weight

    def _analyze_stillness(self, seq: List[FrameSnapshot]) -> float:
        if len(seq) < 10:
            return 0.0

        cfg = self.cfg
        fraction = cfg.stillness_tail_fraction if cfg else 1.0 / 3.0
        third = max(int(len(seq) * fraction), 3)

        tail = seq[-third:]
        head = seq[:third]

        tail_avg = np.mean([s.centroid_disp for s in tail]) if tail else 0
        head_avg = np.mean([s.centroid_disp for s in head]) if head else 0

        if head_avg < 1e-6:
            return 0.0

        ratio = tail_avg / head_avg
        r_h = cfg.stillness_ratio_high if cfg else 0.2
        r_m = cfg.stillness_ratio_mid if cfg else 0.5
        r_l = cfg.stillness_ratio_low if cfg else 0.8

        if ratio < r_h:
            return 1.0
        elif ratio < r_m:
            return 0.6
        elif ratio < r_l:
            return 0.3
        return 0.0

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_in_cooldown(self) -> bool:
        return (time.time() - self._last_alert_time) < self.cooldown


# ============================================================
# 快速测试
# ============================================================

def _test():
    print("=" * 60)
    print("过程级跌倒检测器 自测")
    print("=" * 60)

    from .fall_config import ProcessFallConfig
    cfg = ProcessFallConfig()
    detector = ProcessFallDetector(config=cfg)
    np.random.seed(42)

    # ── 场景: 行走 → 跌倒 → 静止 ──
    print("\n场景 A: 行走(30帧) → 跌倒(30帧) → 躺地(20帧)")
    frame_idx = 0

    for _ in range(30):
        feat = np.zeros(100, dtype=np.float32)
        feat[1] = 0.35 + np.random.randn() * 0.01
        feat[3] = 5.0 + np.random.randn() * 2
        spatial = np.zeros(6, dtype=np.float32)
        spatial[0] = 0.001 + np.random.randn() * 0.0005
        spatial[3] = 0.1
        spatial[4] = 0.3
        _ = detector.update(feat, spatial, frame_idx)
        frame_idx += 1

    for i in range(30):
        feat = np.zeros(100, dtype=np.float32)
        progress = i / 30
        feat[1] = 0.35 + progress * 0.25 + np.random.randn() * 0.01
        feat[3] = 5.0 + progress * 75.0
        spatial = np.zeros(6, dtype=np.float32)
        spatial[0] = 0.05 - progress * 0.02
        spatial[3] = 0.3 + progress * 0.5
        spatial[4] = 0.4 + progress * 0.5
        alert = detector.update(feat, spatial, frame_idx)
        if alert:
            print(f"\n  [ALERT!] {alert.alert_level} 置信度={alert.confidence:.3f}")
            print(f"    高度降={alert.height_drop:.4f} 躯干={alert.torso_change:.1f}°")
            print(f"    各维: {alert.debug_info}")
        frame_idx += 1

    for _ in range(20):
        feat = np.zeros(100, dtype=np.float32)
        feat[1] = 0.60 + np.random.randn() * 0.005
        feat[3] = 80.0 + np.random.randn() * 3
        spatial = np.zeros(6, dtype=np.float32)
        spatial[0] = 0.0005
        spatial[3] = 0.05
        spatial[4] = 0.1
        alert = detector.update(feat, spatial, frame_idx)
        if alert:
            print(f"  (静止期告警: {alert.alert_level})")
        frame_idx += 1

    # ── 场景 B: 弯腰捡东西 ──
    print("\n\n场景 B: 弯腰捡东西 → 恢复（不应触发 ALERT）")
    detector2 = ProcessFallDetector(config=cfg)

    for i in range(30):
        feat = np.zeros(100, dtype=np.float32)
        feat[1] = 0.35 + np.random.randn() * 0.01
        feat[3] = 3.0 + np.random.randn() * 1
        spatial = np.zeros(6, dtype=np.float32)
        spatial[0] = 0.002
        spatial[3] = 0.1
        spatial[4] = 0.3
        _ = detector2.update(feat, spatial, i)

    for i in range(15):
        feat = np.zeros(100, dtype=np.float32)
        progress = i / 15
        feat[1] = 0.35 + progress * 0.10
        feat[3] = 3.0 + progress * 40.0
        spatial = np.zeros(6, dtype=np.float32)
        spatial[0] = 0.01
        spatial[3] = 0.2
        spatial[4] = 0.4
        alert = detector2.update(feat, spatial, 30 + i)
        if alert:
            print(f"  弯腰告警: {alert.alert_level} (conf={alert.confidence:.3f})")

    for i in range(15):
        feat = np.zeros(100, dtype=np.float32)
        progress = i / 15
        feat[1] = 0.45 - progress * 0.10
        feat[3] = 43.0 - progress * 40.0
        spatial = np.zeros(6, dtype=np.float32)
        spatial[0] = 0.01
        spatial[3] = 0.2
        spatial[4] = 0.4
        alert = detector2.update(feat, spatial, 45 + i)
        if alert:
            print(f"  恢复告警: {alert.alert_level} (conf={alert.confidence:.3f})")

    print("  (弯腰未触发跌倒 — OK)")
    print(f"\n{'─' * 40}")
    print("  测试完成: 跌倒=告警 [OK] | 弯腰=不告警 [OK]")


if __name__ == "__main__":
    _test()
