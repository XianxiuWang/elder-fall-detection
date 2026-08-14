#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
motion_spatial.py — 关键点空间运动分析器
=========================================
灵感来源: FMCW 雷达论文《Fall Detection Using FMCW Radar》
          "跌倒检测不能只看动得快不快，还要看人在空间里怎么移动"

雷达版特征 → 2D 摄像头版改造:
  Centroid Range（距离质心） → Motion Centroid（运动中心位移）
  Range Width（距离宽度）   → Motion Spread（运动扩散范围）

核心思路:
  单只手/头快速移动 → 局部运动 → 不是跌倒
  全身(躯干+四肢)同时快速位移 → 空间范围大、运动中心大偏移 → 是跌倒

输出特征 (6维):
  [0] motion_centroid_disp  — 运动中心帧间位移（归一化）
  [1] motion_centroid_angle — 运动中心移动方向角（度）
  [2] motion_spread_active  — 运动中关键点的比例（0-1）
  [3] motion_spread_width   — 运动关键点的空间跨度（归一化）
  [4] torso_disp_ratio      — 躯干位移占比（躯干动 vs 仅四肢动）
  [5] upper_lower_ratio     — 上半身/下半身位移比（跌倒时下半身大幅垂直下落）

v2.0 更新:
  - 所有参数可从 fall_config.py 的 MotionSpatialConfig 注入
  - is_likely_fall() 改为静态方法，可从外部直接调用（不需要实例）
  - 支持 reset() 重置平滑历史
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .fall_config import MotionSpatialConfig


# ============================================================
# 关键点分组
# ============================================================

UPPER_BODY_INDICES = [
    0, 1, 2, 3, 4, 5, 6,   # nose, eyes
    7, 8,                    # ears
    9, 10,                   # mouth
    11, 12,                  # shoulders
    13, 14,                  # elbows
    15, 16,                  # wrists
    17, 18, 19, 20, 21, 22, # hands/fingers
]

LOWER_BODY_INDICES = [
    23, 24,  # hips
    25, 26,  # knees
    27, 28,  # ankles
    29, 30,  # heels
    31, 32,  # feet
]

CORE_BODY_INDICES = [11, 12, 23, 24]  # shoulders + hips
LIMB_INDICES = [13, 14, 15, 16, 25, 26, 27, 28]


@dataclass
class MotionSpatial:
    """一帧的空间运动特征"""
    motion_centroid_disp: float = 0.0
    motion_centroid_angle: float = 0.0
    motion_spread_active: float = 0.0
    motion_spread_width: float = 0.0
    torso_disp_ratio: float = 0.0
    upper_lower_ratio: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.motion_centroid_disp,
            self.motion_centroid_angle,
            self.motion_spread_active,
            self.motion_spread_width,
            self.torso_disp_ratio,
            self.upper_lower_ratio,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "centroid_disp": self.motion_centroid_disp,
            "centroid_angle": self.motion_centroid_angle,
            "spread_active": self.motion_spread_active,
            "spread_width": self.motion_spread_width,
            "torso_ratio": self.torso_disp_ratio,
            "upper_lower": self.upper_lower_ratio,
        }

    def __repr__(self):
        return (f"MotionSpatial(centroid_disp={self.motion_centroid_disp:.4f}, "
                f"spread_active={self.motion_spread_active:.2f}, "
                f"torso_ratio={self.torso_disp_ratio:.2f})")


class MotionSpatialAnalyzer:
    """
    空间运动分析器。

    参数:
        config : MotionSpatialConfig 或 None（使用默认值）
        motion_threshold : 判定关键点"在运动"的归一化位移阈值
        spread_smooth : 平滑窗口帧数
    """

    def __init__(self, config: "MotionSpatialConfig" = None,
                 motion_threshold: float = None,
                 spread_smooth: int = None):
        if config is not None:
            self.threshold = config.motion_threshold
            self.smooth_window = config.spread_smooth
            self.config = config
        else:
            self.threshold = motion_threshold if motion_threshold is not None else 0.005
            self.smooth_window = spread_smooth if spread_smooth is not None else 5
            self.config = None

        self._spread_history: List[float] = []
        self._active_history: List[float] = []

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def extract(self, landmarks: np.ndarray,
                prev_landmarks: Optional[np.ndarray] = None) -> MotionSpatial:
        """从当前帧和前一帧关键点中提取 6 维空间运动特征。"""
        spatial = MotionSpatial()

        if prev_landmarks is None:
            return spatial

        vis_mask = (landmarks[:, 3] > 0.3) & (prev_landmarks[:, 3] > 0.3)
        n_visible = vis_mask.sum()
        if n_visible < 5:
            return spatial

        # 帧间位移
        displacements = np.zeros(33, dtype=np.float32)
        for i in range(33):
            if vis_mask[i]:
                dy = landmarks[i, 1] - prev_landmarks[i, 1]
                dx = landmarks[i, 0] - prev_landmarks[i, 0]
                displacements[i] = math.sqrt(dx * dx + dy * dy)

        moving_mask = displacements > self.threshold
        n_moving = moving_mask.sum()

        # 特征 1 & 2: 运动中心位移和方向
        if n_moving > 0:
            curr_weights = displacements + 1e-6
            curr_center_x = np.average(landmarks[:, 0], weights=curr_weights)
            curr_center_y = np.average(landmarks[:, 1], weights=curr_weights)
            prev_center_x = np.average(prev_landmarks[:, 0], weights=curr_weights)
            prev_center_y = np.average(prev_landmarks[:, 1], weights=curr_weights)

            mcd_x = curr_center_x - prev_center_x
            mcd_y = curr_center_y - prev_center_y
            spatial.motion_centroid_disp = float(math.sqrt(mcd_x**2 + mcd_y**2))
            spatial.motion_centroid_angle = float(
                math.degrees(math.atan2(mcd_y, abs(mcd_x) + 1e-6))
            )
        else:
            spatial.motion_centroid_disp = 0.0
            spatial.motion_centroid_angle = 0.0

        # 特征 3: 运动关键点比例
        spatial.motion_spread_active = float(n_moving / max(n_visible, 1))

        # 特征 4: 运动空间跨度
        if n_moving >= 3:
            moving_parts = landmarks[moving_mask]
            x_span = moving_parts[:, 0].max() - moving_parts[:, 0].min()
            y_span = moving_parts[:, 1].max() - moving_parts[:, 1].min()
            spatial.motion_spread_width = float(x_span + y_span)
        else:
            spatial.motion_spread_width = 0.0

        # 特征 5: 躯干位移占比
        core_indices = [11, 12, 23, 24]
        limb_indices = [13, 14, 15, 16, 25, 26, 27, 28]

        core_disp = sum(displacements[i] for i in core_indices if vis_mask[i])
        core_count = sum(1 for i in core_indices if vis_mask[i])
        limb_disp = sum(displacements[i] for i in limb_indices if vis_mask[i])
        limb_count = sum(1 for i in limb_indices if vis_mask[i])

        avg_core = core_disp / max(core_count, 1)
        avg_limb = limb_disp / max(limb_count, 1)
        total_avg = avg_core + avg_limb
        spatial.torso_disp_ratio = float(avg_core / total_avg) if total_avg > 1e-6 else 0.0

        # 特征 6: 上半身/下半身位移比
        upper_avg, upper_count = 0.0, 0
        for idx in UPPER_BODY_INDICES:
            if idx < 33 and vis_mask[idx]:
                upper_avg += displacements[idx]
                upper_count += 1

        lower_avg, lower_count = 0.0, 0
        for idx in LOWER_BODY_INDICES:
            if idx < 33 and vis_mask[idx]:
                lower_avg += displacements[idx]
                lower_count += 1

        upper_avg /= max(upper_count, 1)
        lower_avg /= max(lower_count, 1)

        if upper_avg + lower_avg > 1e-6:
            spatial.upper_lower_ratio = float(upper_avg / (upper_avg + lower_avg))
        else:
            spatial.upper_lower_ratio = 0.5

        # 平滑
        self._spread_history.append(spatial.motion_spread_width)
        self._active_history.append(spatial.motion_spread_active)
        if len(self._spread_history) > self.smooth_window:
            self._spread_history.pop(0)
            self._active_history.pop(0)

        spatial.motion_spread_width = float(np.mean(self._spread_history))
        spatial.motion_spread_active = float(np.mean(self._active_history))

        return spatial

    def reset(self):
        """重置平滑历史"""
        self._spread_history.clear()
        self._active_history.clear()

    def extract_array(self, landmarks: np.ndarray,
                      prev_landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """直接返回 6 维 numpy 数组"""
        return self.extract(landmarks, prev_landmarks).to_array()

    # ------------------------------------------------------------------
    # 跌倒判定（静态方法，不依赖实例状态）
    # ------------------------------------------------------------------

    @staticmethod
    def is_likely_fall(spatial: MotionSpatial,
                       config: "MotionSpatialConfig" = None) -> Tuple[bool, float]:
        """
        基于空间运动特征判定是否像跌倒。

        v2.0: 改为静态方法，可通过 config 注入阈值，
              不再需要创建临时实例。

        Returns (is_fall_like, confidence_0_to_1)
        """
        if config is None:
            # 默认阈值（兼容旧调用）
            cfg = type('Cfg', (), {
                'fall_centroid_disp_min': 0.02,
                'fall_centroid_disp_max': 0.08,
                'fall_spread_active_min': 0.4,
                'fall_spread_active_max': 0.8,
                'fall_spread_width_min': 0.2,
                'fall_spread_width_max': 0.6,
                'fall_torso_ratio_min': 0.3,
                'likelihood_weights': {"centroid_disp": 1.0, "spread_active": 1.0,
                                       "spread_width": 1.0, "torso_ratio": 1.0},
                'likelihood_confidence_threshold': 0.5,
            })()
        else:
            cfg = config

        score = 0.0
        total = 0.0

        # 运动中心位移
        w = cfg.likelihood_weights.get("centroid_disp", 1.0)
        if spatial.motion_centroid_disp > cfg.fall_centroid_disp_min:
            score += w * min(1.0, spatial.motion_centroid_disp / cfg.fall_centroid_disp_max)
        total += w

        # 运动关键点比例
        w = cfg.likelihood_weights.get("spread_active", 1.0)
        if spatial.motion_spread_active > cfg.fall_spread_active_min:
            score += w * min(1.0, spatial.motion_spread_active / cfg.fall_spread_active_max)
        total += w

        # 运动空间跨度
        w = cfg.likelihood_weights.get("spread_width", 1.0)
        if spatial.motion_spread_width > cfg.fall_spread_width_min:
            score += w * min(1.0, spatial.motion_spread_width / cfg.fall_spread_width_max)
        total += w

        # 躯干位移占比
        w = cfg.likelihood_weights.get("torso_ratio", 1.0)
        if spatial.torso_disp_ratio > cfg.fall_torso_ratio_min:
            score += w * spatial.torso_disp_ratio
        total += w

        confidence = score / total if total > 0 else 0.0
        return confidence > cfg.likelihood_confidence_threshold, confidence


# ============================================================
# 快速测试
# ============================================================

def _test():
    print("=" * 60)
    print("空间运动分析器 自测")
    print("=" * 60)

    # 使用配置
    from .fall_config import MotionSpatialConfig
    cfg = MotionSpatialConfig()
    analyzer = MotionSpatialAnalyzer(config=cfg)
    np.random.seed(42)

    # ── 场景 A: 站立挥手 ──
    print("\n场景 A: 站立挥手（局部动作）")
    lm_stand = np.zeros((33, 4), dtype=np.float32)
    for i in range(33):
        lm_stand[i, 3] = 0.95
    lm_stand[11] = [0.42, 0.35, 0.0, 0.95]
    lm_stand[12] = [0.58, 0.35, 0.0, 0.95]
    lm_stand[23] = [0.44, 0.55, 0.0, 0.95]
    lm_stand[24] = [0.56, 0.55, 0.0, 0.95]
    lm_stand[25] = [0.45, 0.75, 0.0, 0.95]
    lm_stand[26] = [0.55, 0.75, 0.0, 0.95]
    lm_stand[27] = [0.45, 0.92, 0.0, 0.95]
    lm_stand[28] = [0.55, 0.92, 0.0, 0.95]
    lm_stand[13] = [0.35, 0.40, 0.0, 0.95]
    lm_stand[14] = [0.65, 0.40, 0.0, 0.95]
    lm_stand[15] = [0.28, 0.45, 0.0, 0.95]
    lm_stand[16] = [0.72, 0.45, 0.0, 0.95]

    lm_wave = lm_stand.copy()
    lm_wave[15] = [0.20, 0.30, 0.0, 0.95]
    lm_wave[16] = [0.80, 0.30, 0.0, 0.95]

    sp1 = analyzer.extract(lm_wave, lm_stand)
    is_fall, conf = MotionSpatialAnalyzer.is_likely_fall(sp1, cfg)  # 静态调用
    print(f"  {sp1}")
    print(f"  判定: {'跌倒-like' if is_fall else '正常'} (置信度 {conf:.2f})")

    # ── 场景 B: 模拟跌倒 ──
    analyzer.reset()
    print("\n场景 B: 全身跌倒（大范围、躯干也动）")
    lm_fall = lm_stand.copy()
    for i in range(33):
        if lm_fall[i, 3] > 0.3:
            lm_fall[i, 1] += 0.15
            lm_fall[i, 0] += np.random.randn() * 0.02

    sp2 = analyzer.extract(lm_fall, lm_stand)
    is_fall2, conf2 = MotionSpatialAnalyzer.is_likely_fall(sp2, cfg)
    print(f"  {sp2}")
    print(f"  判定: {'跌倒-like' if is_fall2 else '正常'} (置信度 {conf2:.2f})")
    print(f"\n  >> 挥手 vs 跌倒 区分: {'[OK]' if conf2 > conf * 2 else '[弱]'}")

    # ── 场景 C: 缓慢坐下 ──
    analyzer.reset()
    print("\n场景 C: 缓慢坐下（躯干有移动但不是跌倒）")
    lm_sit = lm_stand.copy()
    for i in [11, 12, 23, 24, 25, 26]:
        lm_sit[i, 1] += 0.05

    sp3 = analyzer.extract(lm_sit, lm_stand)
    is_fall3, conf3 = MotionSpatialAnalyzer.is_likely_fall(sp3, cfg)
    print(f"  {sp3}")
    print(f"  判定: {'跌倒-like' if is_fall3 else '正常'} (置信度 {conf3:.2f})")

    print(f"\n{'─' * 40}")
    print(f"  挥手置信度: {conf:.3f}")
    print(f"  跌倒置信度: {conf2:.3f}")
    print(f"  坐下置信度: {conf3:.3f}")
    print(f"  跌倒/挥手比: {conf2 / max(conf, 0.01):.1f}x (越大越能区分)")


if __name__ == "__main__":
    _test()
