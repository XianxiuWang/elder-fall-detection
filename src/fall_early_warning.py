#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fall_early_warning.py — 跌倒预测/早期预警系统 v1.0
==================================================
与 FallPredictor (检测已发生的跌倒) 不同，本模块聚焦于**预测**：
在摔倒发生前 1-3 秒，通过步态不稳定性趋势分析提前预警。

原理:
  从 33 个 MediaPipe 关键点提取 6 个"预跌倒"生物力学指标，
  滑动窗口持续追踪每个指标的趋势线和加速度，
  当 ≥3 个指标同时恶化 → 触发"跌倒预警"。

六指标:
  1. COM 横向摇摆加速度     — 平衡恶化
  2. 步宽变异度             — 站立不稳加剧
  3. 躯干前倾角速度         — 即将倾倒
  4. 支撑面缩小率           — 基底支撑崩溃
  5. 手臂"扑腾"度           — 试图恢复平衡
  6. 步态节律紊乱           — 行走节奏破碎

用法:
    early_warn = FallEarlyWarning(fps=15.0)
    report = early_warn.update(landmarks_33x4, elapsed_sec, is_walking)
    if report.pre_fall_risk >= 60:
        print(f"[预 警] 跌倒风险升高！(得分: {report.pre_fall_risk})")
"""

import os, sys, time
from typing import Optional, Dict, Deque, List, Tuple
from collections import deque
from dataclasses import dataclass, field
import numpy as np

# ── MediaPipe 33关键点索引 (与 fall_predictor.py 一致) ──
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32


@dataclass
class EarlyWarningReport:
    """跌倒早期预警报告"""

    # ═══ 综合预测 ═══
    pre_fall_risk: float = 0.0          # 0-100 预测风险分
    alert_level: str = "SAFE"           # SAFE / WATCH / WARNING / CRITICAL
    alert_level_cn: str = "安全"
    indicators_active: int = 0          # 当前活跃的恶化指标数
    indicators_triggered: List[str] = field(default_factory=list)

    # ═══ 六指标分项风险 (0-100) ═══
    sway_accel_risk: float = 0.0        # COM 横向摇摆加速度风险
    step_width_risk: float = 0.0        # 步宽变异度风险
    trunk_tilt_vel_risk: float = 0.0    # 躯干倾斜角速度风险
    support_shrink_risk: float = 0.0    # 支撑面缩小风险
    arm_flail_risk: float = 0.0         # 手臂扑腾风险
    gait_rhythm_risk: float = 0.0       # 步态节律紊乱风险

    # ═══ 原始测量值 ═══
    sway_accel: float = 0.0             # 摇摆加速度
    step_width_cv: float = 0.0          # 步宽变异系数
    trunk_tilt_vel: float = 0.0         # 躯干倾斜角速度 (deg/s)
    support_area_rate: float = 0.0      # 支撑面变化率
    wrist_speed_max: float = 0.0        # 最大腕部速度
    gait_cv: float = 0.0               # 步态节律变异系数

    # ═══ 建议 ═══
    suggestion: str = ""
    timestamp: float = 0.0


class FallEarlyWarning:
    """跌倒早期预警系统 v1.0

    设计理念:
      不等摔倒，追踪"失衡恶化过程"。当步态稳定性特征出现
      多维度同步退化时，提前拉响预警。
    """

    # ── 窗口参数 ──
    HISTORY_SEC = 6.0            # 历史窗口 (跟踪趋势线)
    TREND_SEC = 2.0              # 趋势计算窗口
    ACCEL_SEC = 1.0              # 加速度计算窗口

    # ── 指标触发阈值 ──
    # 1. COM 横向摇摆加速度
    SWAY_ACCEL_WARN = 0.004      # 归一化加速度阈值
    SWAY_ACCEL_DANGER = 0.008

    # 2. 步宽变异度 (CV)
    STEP_WIDTH_CV_WARN = 0.15    # 变异系数 > 15%
    STEP_WIDTH_CV_DANGER = 0.25

    # 3. 躯干前倾角速度
    TRUNK_TILT_VEL_WARN = 15.0   # deg/s
    TRUNK_TILT_VEL_DANGER = 30.0

    # 4. 支撑面缩小率 (面积/秒)
    SUPPORT_SHRINK_WARN = 0.15   # 15% per second
    SUPPORT_SHRINK_DANGER = 0.30

    # 5. 手臂扑腾度 (归一化速度)
    WRIST_SPEED_WARN = 0.08
    WRIST_SPEED_DANGER = 0.15

    # 6. 步态节律变异系数
    GAIT_CV_WARN = 0.25
    GAIT_CV_DANGER = 0.40

    # ── 投票阈值 ──
    INDICATORS_WATCH = 1         # 1个触发 → 关注
    INDICATORS_WARNING = 2       # 2个触发 → 预警
    INDICATORS_CRITICAL = 3      # 3个触发 → 严重预警

    # ── 预测风险分阈值 ──
    RISK_WATCH = 30.0
    RISK_WARNING = 50.0
    RISK_CRITICAL = 70.0

    def __init__(self, fps: float = 15.0):
        self.fps = max(fps, 1.0)
        self._dt = 1.0 / self.fps

        # 窗口大小
        self._hist_size = max(int(self.HISTORY_SEC * self.fps), 20)
        self._trend_size = max(int(self.TREND_SEC * self.fps), 10)
        self._accel_size = max(int(self.ACCEL_SEC * self.fps), 5)

        # ── 指标历史缓冲区 ──
        self._sway_history: Deque[float] = deque(maxlen=self._hist_size)
        self._step_width_history: Deque[float] = deque(maxlen=self._hist_size)
        self._trunk_tilt_history: Deque[float] = deque(maxlen=self._hist_size)
        self._support_area_history: Deque[float] = deque(maxlen=self._hist_size)
        self._wrist_speed_history: Deque[float] = deque(maxlen=self._hist_size)

        # ── 步态节律追踪 ──
        self._step_times: Deque[float] = deque(maxlen=30)
        self._step_intervals: Deque[float] = deque(maxlen=30)

        # ── 上一帧状态 ──
        self._prev_com_x: Optional[float] = None
        self._prev_sway_vel: Optional[float] = None
        self._prev_trunk_tilt: Optional[float] = None
        self._prev_wrist_L: Optional[np.ndarray] = None
        self._prev_wrist_R: Optional[np.ndarray] = None
        self._prev_support_area: Optional[float] = None

        # ── EMA 平滑 ──
        self._ema: Dict[str, float] = {}

    # ═══════════════════════════════════════════════
    # 主接口
    # ═══════════════════════════════════════════════

    def update(self, landmarks: np.ndarray, elapsed: float,
               is_walking: bool = True) -> EarlyWarningReport:
        """
        Args:
            landmarks: (33, 4) keypoints (x, y, z, visibility)
            elapsed: 会话运行时间 (秒)
            is_walking: 当前是否在行走 (非行走时抑制部分指标)
        Returns:
            EarlyWarningReport
        """
        # ═══ 1. 计算六指标 ═══
        sway_accel = self._compute_sway_acceleration(landmarks, is_walking)
        step_width_cv = self._compute_step_width_cv(landmarks, is_walking)
        trunk_tilt_vel = self._compute_trunk_tilt_velocity(landmarks, is_walking)
        support_shrink = self._compute_support_shrink(landmarks, is_walking)
        arm_flail = self._compute_arm_flailing(landmarks, is_walking)
        gait_rhythm_cv, _ = self._compute_gait_rhythm(landmarks, elapsed, is_walking)

        # ═══ 2. 每个指标 → 风险分 ═══
        sway_accel_risk = self._score_indicator(
            sway_accel, self.SWAY_ACCEL_WARN, self.SWAY_ACCEL_DANGER,
            "sway_accel", is_walking)
        step_width_risk = self._score_indicator(
            step_width_cv, self.STEP_WIDTH_CV_WARN, self.STEP_WIDTH_CV_DANGER,
            "step_width", is_walking)
        trunk_tilt_vel_risk = self._score_indicator(
            trunk_tilt_vel, self.TRUNK_TILT_VEL_WARN, self.TRUNK_TILT_VEL_DANGER,
            "trunk_tilt", is_walking)
        support_shrink_risk = self._score_indicator(
            support_shrink, self.SUPPORT_SHRINK_WARN, self.SUPPORT_SHRINK_DANGER,
            "support_shrink", is_walking)
        arm_flail_risk = self._score_indicator(
            arm_flail, self.WRIST_SPEED_WARN, self.WRIST_SPEED_DANGER,
            "arm_flail", is_walking)
        gait_rhythm_risk = self._score_indicator(
            gait_rhythm_cv, self.GAIT_CV_WARN, self.GAIT_CV_DANGER,
            "gait_rhythm", is_walking)

        # ═══ 3. 投票：多少个指标越过 WARNING 线 ═══
        indicators = [
            (sway_accel_risk, "COM摇摆加速"),
            (step_width_risk, "步宽变异"),
            (trunk_tilt_vel_risk, "躯干倾斜速度"),
            (support_shrink_risk, "支撑面缩小"),
            (arm_flail_risk, "手臂扑腾"),
            (gait_rhythm_risk, "步态节律"),
        ]
        triggered = [(name, risk) for risk, name in indicators if risk >= 50.0]
        active_count = len(triggered)

        # ═══ 4. 综合预测风险分 ═══
        # 加权融合：激活的指标越多，合成风险越高
        all_risks = [sway_accel_risk, step_width_risk, trunk_tilt_vel_risk,
                      support_shrink_risk, arm_flail_risk, gait_rhythm_risk]
        # 取 top-N 均值 (越少指标激活，分越低)
        top_n = max(2, active_count + 1)
        top_risks = sorted(all_risks, reverse=True)[:top_n]
        # 非线性加成：≥3 激活时额外加 20%
        base_risk = np.mean(top_risks) if top_risks else 0.0
        if active_count >= self.INDICATORS_CRITICAL:
            base_risk = base_risk * 1.2 + 10.0
        elif active_count >= self.INDICATORS_WARNING:
            base_risk = base_risk * 1.1
        pre_fall_risk = max(0.0, min(100.0, base_risk))

        # ═══ 5. 告警等级 ═══
        if pre_fall_risk >= self.RISK_CRITICAL and active_count >= self.INDICATORS_CRITICAL:
            alert_level, alert_cn = "CRITICAL", "严重预警"
            suggestion = "跌倒风险极高！多个平衡指标同步恶化，请立即关注老人！"
        elif pre_fall_risk >= self.RISK_WARNING and active_count >= self.INDICATORS_WARNING:
            alert_level, alert_cn = "WARNING", "预警"
            triggered_str = "、".join([t[0] for t in triggered])
            suggestion = f"跌倒风险升高！触发指标: {triggered_str}，注意观察步态稳定性"
        elif pre_fall_risk >= self.RISK_WATCH or active_count >= self.INDICATORS_WATCH:
            alert_level, alert_cn = "WATCH", "关注"
            suggestion = "步态存在轻微不稳迹象，保持观察"
        else:
            alert_level, alert_cn = "SAFE", "安全"
            suggestion = ""

        return EarlyWarningReport(
            pre_fall_risk=round(pre_fall_risk, 1),
            alert_level=alert_level,
            alert_level_cn=alert_cn,
            indicators_active=active_count,
            indicators_triggered=[t[0] for t in triggered],
            sway_accel_risk=round(sway_accel_risk, 1),
            step_width_risk=round(step_width_risk, 1),
            trunk_tilt_vel_risk=round(trunk_tilt_vel_risk, 1),
            support_shrink_risk=round(support_shrink_risk, 1),
            arm_flail_risk=round(arm_flail_risk, 1),
            gait_rhythm_risk=round(gait_rhythm_risk, 1),
            sway_accel=round(sway_accel, 6),
            step_width_cv=round(step_width_cv, 4),
            trunk_tilt_vel=round(trunk_tilt_vel, 2),
            support_area_rate=round(support_shrink, 4),
            wrist_speed_max=round(arm_flail, 4),
            gait_cv=round(gait_rhythm_cv, 4),
            suggestion=suggestion,
            timestamp=elapsed,
        )

    # ═══════════════════════════════════════════════
    # 六指标计算
    # ═══════════════════════════════════════════════

    def _compute_sway_acceleration(self, lm: np.ndarray,
                                   is_walking: bool) -> float:
        """COM 横向摇摆加速度 (归一化)"""
        shoulder_mid_x = (lm[LEFT_SHOULDER, 0] + lm[RIGHT_SHOULDER, 0]) / 2
        hip_mid_x = (lm[LEFT_HIP, 0] + lm[RIGHT_HIP, 0]) / 2
        # 躯干横向偏移
        sway = abs(shoulder_mid_x - hip_mid_x)
        self._sway_history.append(sway)

        if self._prev_sway_vel is None:
            self._prev_sway_vel = 0.0
            return 0.0

        # 加速度 = 摇摆速度的变化率
        sway_vel = sway - (self._sway_history[-2] if len(self._sway_history) > 1 else sway)
        sway_vel = abs(sway_vel)
        accel = abs(sway_vel - self._prev_sway_vel)
        self._prev_sway_vel = sway_vel

        if not is_walking:
            accel *= 0.3  # 静止时降低敏感度
        return self._ema_val("sway_accel", accel, 0.3)

    def _compute_step_width_cv(self, lm: np.ndarray,
                                is_walking: bool) -> float:
        """步宽变异度：双踝距离的变异系数"""
        ankle_dist = abs(lm[LEFT_ANKLE, 0] - lm[RIGHT_ANKLE, 0])
        # 归一化：除以髋宽
        hip_width = max(abs(lm[LEFT_HIP, 0] - lm[RIGHT_HIP, 0]), 0.01)
        normalized_dist = ankle_dist / hip_width
        self._step_width_history.append(normalized_dist)

        if len(self._step_width_history) < self._trend_size:
            return 0.0

        recent = list(self._step_width_history)[-self._trend_size:]
        recent_arr = np.array(recent)
        mean_val = np.mean(recent_arr)
        std_val = np.std(recent_arr)
        cv = std_val / mean_val if mean_val > 0.001 else 0.0

        if not is_walking:
            cv *= 0.5
        return self._ema_val("step_width_cv", cv, 0.2)

    def _compute_trunk_tilt_velocity(self, lm: np.ndarray,
                                      is_walking: bool) -> float:
        """躯干前倾角速度 (度/秒)"""
        shoulder_mid = (lm[LEFT_SHOULDER, :2] + lm[RIGHT_SHOULDER, :2]) / 2
        hip_mid = (lm[LEFT_HIP, :2] + lm[RIGHT_HIP, :2]) / 2
        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]
        if abs(dy) < 1e-6:
            tilt = 0.0
        else:
            tilt = float(np.degrees(np.arctan2(abs(dx), abs(dy))))
        self._trunk_tilt_history.append(tilt)

        if self._prev_trunk_tilt is None:
            self._prev_trunk_tilt = tilt
            return 0.0

        # 角速度 (°/frame) → (°/s)
        tilt_diff = abs(tilt - self._prev_trunk_tilt)
        angular_vel = tilt_diff / self._dt
        self._prev_trunk_tilt = tilt

        if not is_walking:
            angular_vel *= 0.4
        return self._ema_val("trunk_tilt_vel", angular_vel, 0.25)

    def _compute_support_shrink(self, lm: np.ndarray,
                                 is_walking: bool) -> float:
        """支撑面缩小率：双足包围盒面积变化"""
        # 双足边界
        foot_x = [lm[LEFT_ANKLE, 0], lm[RIGHT_ANKLE, 0],
                   lm[LEFT_HEEL, 0], lm[RIGHT_HEEL, 0],
                   lm[LEFT_FOOT_INDEX, 0], lm[RIGHT_FOOT_INDEX, 0]]
        foot_y = [lm[LEFT_ANKLE, 1], lm[RIGHT_ANKLE, 1],
                   lm[LEFT_HEEL, 1], lm[RIGHT_HEEL, 1],
                   lm[LEFT_FOOT_INDEX, 1], lm[RIGHT_FOOT_INDEX, 1]]
        width = max(foot_x) - min(foot_x)
        height = max(foot_y) - min(foot_y)
        area = max(width * height, 1e-8)
        self._support_area_history.append(area)

        if self._prev_support_area is None:
            self._prev_support_area = area
            return 0.0

        # 面积变化率
        if self._prev_support_area < 1e-8:
            shrink_rate = 0.0
        else:
            shrink_rate = max(0.0, (self._prev_support_area - area) / self._prev_support_area)
        self._prev_support_area = area

        if not is_walking:
            shrink_rate *= 0.2
        return self._ema_val("support_shrink", shrink_rate, 0.2)

    def _compute_arm_flailing(self, lm: np.ndarray,
                               is_walking: bool) -> float:
        """手臂扑腾度：腕部突发速度峰值"""
        wrist_L = np.array([lm[LEFT_WRIST, 0], lm[LEFT_WRIST, 1]])
        wrist_R = np.array([lm[RIGHT_WRIST, 0], lm[RIGHT_WRIST, 1]])

        if self._prev_wrist_L is None:
            self._prev_wrist_L = wrist_L
            self._prev_wrist_R = wrist_R
            return 0.0

        # 腕部移动速度（归一化，取左右最大值）
        vel_L = float(np.linalg.norm(wrist_L - self._prev_wrist_L))
        vel_R = float(np.linalg.norm(wrist_R - self._prev_wrist_R))
        max_wrist_speed = max(vel_L, vel_R)
        self._wrist_speed_history.append(max_wrist_speed)

        self._prev_wrist_L = wrist_L.copy()
        self._prev_wrist_R = wrist_R.copy()

        # 取最近窗口的峰值速度（扑腾=突发高速度）
        if len(self._wrist_speed_history) < self._accel_size:
            peak = max_wrist_speed
        else:
            recent_speeds = list(self._wrist_speed_history)[-self._accel_size:]
            avg_speed = np.mean(recent_speeds)
            peak = max_wrist_speed - avg_speed  # 超出均值的部分
            peak = max(0.0, peak)

        if not is_walking:
            peak *= 0.3
        return self._ema_val("arm_flail", peak, 0.35)

    def _compute_gait_rhythm(self, lm: np.ndarray, elapsed: float,
                              is_walking: bool) -> Tuple[float, bool]:
        """步态节律变异系数 + 是否检测到步伐"""
        # 简化的步态检测：踝部 Y 坐标过零点
        ankle_y = (lm[LEFT_ANKLE, 1] + lm[RIGHT_ANKLE, 1]) / 2
        step_detected = False

        # 跟踪踝部 Y 坐标的局部极值
        if hasattr(self, '_ankle_y_vals'):
            self._ankle_y_vals.append(ankle_y)
            if len(self._ankle_y_vals) >= 5:
                recent = list(self._ankle_y_vals)
                # 检查是否穿过零点 (相对均值)
                mean_y = np.mean(recent)
                if len(recent) >= 2:
                    prev_above = (recent[-2] - mean_y) > 0.002
                    curr_above = (recent[-1] - mean_y) > 0.002
                    if prev_above != curr_above:
                        step_detected = True
                        self._step_times.append(elapsed)
                        if len(self._step_times) >= 2:
                            interval = self._step_times[-1] - self._step_times[-2]
                            if 0.3 < interval < 2.0:
                                self._step_intervals.append(interval)
        else:
            self._ankle_y_vals = deque(maxlen=10)
            self._ankle_y_vals.append(ankle_y)

        # 计算步态节律 CV
        if len(self._step_intervals) < 3:
            return 0.0, step_detected

        intervals_arr = np.array(list(self._step_intervals)[-10:])
        mean_interval = np.mean(intervals_arr)
        if mean_interval < 0.01:
            return 0.0, step_detected
        cv = float(np.std(intervals_arr) / mean_interval)

        if not is_walking:
            cv *= 0.4
        return self._ema_val("gait_rhythm", cv, 0.3), step_detected

    # ═══════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════

    def _score_indicator(self, value: float, warn_thresh: float,
                          danger_thresh: float, name: str,
                          is_walking: bool) -> float:
        """将指标值映射到 0-100 风险分"""
        if not is_walking:
            # 非行走时压缩风险分（静止状态下指标波动不代表跌倒风险）
            value = value * 0.5

        if value >= danger_thresh:
            return 70.0 + min(30.0, (value - danger_thresh) / danger_thresh * 30.0)
        elif value >= warn_thresh:
            return 30.0 + (value - warn_thresh) / (danger_thresh - warn_thresh) * 40.0
        elif value >= warn_thresh * 0.5:
            return (value - warn_thresh * 0.5) / (warn_thresh * 0.5) * 30.0
        else:
            return 0.0

    def _ema_val(self, key: str, value: float, alpha: float) -> float:
        if key not in self._ema:
            self._ema[key] = value
        else:
            self._ema[key] = alpha * value + (1 - alpha) * self._ema[key]
        return self._ema[key]


# ═══════════════════════════════════════════════
# 测试入口
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    warn = FallEarlyWarning(fps=15.0)

    print("=" * 60)
    print("  Fall Early Warning — 跌倒预测模块测试")
    print("=" * 60)
    print("  六指标实时追踪:")
    print("    1. COM 横向摇摆加速度")
    print("    2. 步宽变异度")
    print("    3. 躯干前倾角速度")
    print("    4. 支撑面缩小率")
    print("    5. 手臂扑腾度")
    print("    6. 步态节律紊乱")
    print("=" * 60)
    print("  按 Q 退出")
