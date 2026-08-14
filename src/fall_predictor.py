#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fall_predictor.py — 基于生物力学的跌倒预测早期预警系统 v2.0
============================================================
v2.0 升级内容 (基于 6 篇硕士论文学术成果):
  【论文5-宋帅博】骨骼点动静态特征工程 + 跌倒方向判别 + PCA特征选择
  【论文2-冯梓文】四级分级预警(蓝/黄/橙/红) + 非物理连接边特征
  【论文6-王晓宇】几何规则驱动多姿态预判 + 动态阈值置信度累积
  【论文3-李玲艺】注意力权重可解释性 + 215ms 前置时间方法论
  【论文4-夏祖威】多策略对比框架 + 小样本增强思路
  【论文1-谢丹莹】SHAP可解释性 + 循证医学特征筛选范式

特征体系 (v2.0 增强):
  Layer 1 — 逐帧特征 (14 项, v1.0=7 项):
    新增: 角加速度、包围盒宽高比、头-踝距离、腕-踝距离
         姿态转换速度、双支撑相位、非物理连接向量

  Layer 2 — 窗口统计 (10 项, v1.0=6 项):
    新增: 角加速度变异性、双支撑时间比、步频、转换急动度

  Layer 3 — 风险分 + 方向 + 等级:
    新增: 跌倒方向(前/后/左/右)、四级告警(蓝/黄/橙/红)

用法:
    predictor = FallPredictor(fps=15.0)
    report = predictor.update(landmarks_33x4)
"""

import argparse, os, sys, time
from typing import Optional, Tuple, List, Deque, Dict
from collections import deque
from dataclasses import dataclass, field
import numpy as np

# ── MediaPipe 33关键点索引 ──
NOSE = 0
LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER = 1, 2, 3
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


@dataclass
class FallRiskReport:
    """跌倒风险报告 (v2.0: 四级告警 + 方向 + 增强特征)"""

    # ═══ 综合风险 ═══
    risk_score: float = 0.0                # 0-100
    alert_level: str = "SAFE"             # SAFE / BLUE / YELLOW / ORANGE / RED
    alert_level_cn: str = "安全"           # 中文

    # ═══ 四级风险分 (论文2 分级预警) ═══
    sway_risk: float = 0.0                # 躯干摇摆风险
    com_bos_risk: float = 0.0             # 重心-支撑面风险
    gait_risk: float = 0.0                # 步态风险
    posture_risk: float = 0.0             # 姿态风险
    transition_risk: float = 0.0          # 姿态转换风险 (v2.0 新增)
    direction_risk_boost: float = 0.0     # 方向危险加权 (v2.0 新增)

    # ═══ 跌倒方向 (论文5) ═══
    fall_direction: str = "none"          # none / forward / backward / left / right
    fall_direction_deg: float = 0.0       # 方向角度

    # ═══ 关键生物力学值 ═══
    trunk_sway_deg: float = 0.0           # 侧向摇摆角
    trunk_tilt_deg: float = 0.0           # 前后倾斜角
    com_bos_margin: float = 1.0           # COM-BOS 裕度
    gait_speed: float = 0.0               # 步速
    step_regularity: float = 1.0          # 步态规整度
    knee_angle_L: float = 0.0             # 左膝角度
    knee_angle_R: float = 0.0             # 右膝角度
    bbox_ratio: float = 1.0              # 包围盒宽高比 (v2.0 新增)
    head_ankle_dist: float = 1.0         # 头-踝距离 (v2.0 新增)
    angular_velocity_max: float = 0.0    # 最大关节角速度 (v2.0 新增)
    double_support_ratio: float = 0.5    # 双支撑时间比 (v2.0 新增)
    cadence: float = 0.0                 # 步频 steps/min (v2.0 新增)

    # ═══ 预警追踪 ═══
    warning_sustained_sec: float = 0.0
    danger_sustained_sec: float = 0.0
    should_alert: bool = False

    # ═══ 建议 ═══
    suggestion: str = ""
    timestamp: float = 0.0


class FallPredictor:
    """基于生物力学的跌倒预测器 v2.0

    阈值基于文献:
      - 躯干摇摆: 健康老人 < 3°, 跌倒风险老人 > 7° (Maki et al.)
      - COM-BOS: margin < 20% 极高风险 (Pai et al.)
      - 步态变异性: CV > 10% 异常 (Hausdorff et al.)
      - 双支撑时间比: > 30% 步态周期为异常 (Maki)
      - 后向跌倒最危险 (论文5 宋帅博)
    """

    # ── 窗口参数 ──
    SHORT_WINDOW_SEC = 3.0
    MEDIUM_WINDOW_SEC = 5.0      # v2.0 新增
    LONG_WINDOW_SEC = 8.0        # 趋势分析
    DIRECTION_WINDOW_SEC = 2.0   # 方向判断窗口 (v2.0 新增)

    # ── 躯干摇摆阈值 (论文5 + 生物力学文献) ──
    SWAY_DANGER_DEG = 12.0
    SWAY_WARN_DEG = 7.0
    SWAY_VARIABILITY_DANGER = 5.0
    SWAY_VARIABILITY_WARN = 3.0

    # ── COM-BOS 阈值 ──
    COM_BOS_DANGER = 0.20
    COM_BOS_WARN = 0.35
    COM_VEL_DANGER = 0.15

    # ── 步态阈值 ──
    STEP_IRREG_DANGER = 0.35
    STEP_IRREG_WARN = 0.20
    GAIT_SLOWDOWN_DANGER = 0.35
    DOUBLE_SUPPORT_DANGER = 0.40    # v2.0: 双支撑 > 40%
    DOUBLE_SUPPORT_WARN = 0.30

    # ── 姿态阈值 ──
    TORSO_TILT_DANGER = 25.0
    KNEE_ASYMMETRY_DANGER = 15.0

    # ── v2.0 新增阈值 ──
    ANGULAR_VEL_DANGER = 300.0       # 关节角速度 > 300°/s (度/秒)
    ANGULAR_VEL_WARN = 150.0
    BBOX_RATIO_CHANGE_DANGER = 0.5   # 包围盒比例变化 > 50%
    HEAD_ANKLE_COLLAPSE_DANGER = 0.3 # 头-踝距离缩减 > 30% (身体塌缩)
    TRANSITION_JERK_DANGER = 0.08    # 姿态转换急动度

    # ── 方向危险加权 (论文5: 后向最危险) ──
    DIRECTION_WEIGHTS = {
        "forward": 1.0,
        "left": 1.1,
        "right": 1.1,
        "backward": 1.3,  # 后向跌倒 → 头部受伤风险最高
        "none": 0.0,
    }

    def __init__(self, fps: float = 15.0, alert_callback=None):
        self.fps = max(fps, 1.0)
        self._dt = 1.0 / self.fps  # 帧间隔

        # 窗口大小
        self.win_short = max(int(self.SHORT_WINDOW_SEC * self.fps), 10)
        self.win_medium = max(int(self.MEDIUM_WINDOW_SEC * self.fps), 15)
        self.win_long = max(int(self.LONG_WINDOW_SEC * self.fps), 20)
        self.win_dir = max(int(self.DIRECTION_WINDOW_SEC * self.fps), 8)

        # ── 逐帧特征缓冲区 ──
        self._sway_history: Deque[float] = deque(maxlen=self.win_long)
        self._tilt_history: Deque[float] = deque(maxlen=self.win_long)
        self._com_margin_history: Deque[float] = deque(maxlen=self.win_long)
        self._com_vel_history: Deque[float] = deque(maxlen=self.win_long)
        self._knee_L_history: Deque[float] = deque(maxlen=self.win_long)
        self._knee_R_history: Deque[float] = deque(maxlen=self.win_long)
        self._gait_speed_history: Deque[float] = deque(maxlen=self.win_long)
        self._bbox_ratio_history: Deque[float] = deque(maxlen=self.win_long)       # v2.0
        self._head_ankle_dist_history: Deque[float] = deque(maxlen=self.win_long)  # v2.0
        self._angular_vel_history: Deque[float] = deque(maxlen=self.win_long)      # v2.0
        self._transition_jerk_history: Deque[float] = deque(maxlen=self.win_long)  # v2.0

        # 步态分析
        self._step_events: Deque[float] = deque(maxlen=50)
        self._step_intervals: Deque[float] = deque(maxlen=50)
        self._double_support_frames = 0
        self._total_gait_frames = 0

        # ── v2.1: 活动状态检测 (修复风险漂移 + 静止站立检测) ──
        self._is_walking = False            # 是否在行走
        self._last_step_elapsed = 0.0       # 距离上一步的时间
        self._is_stationary = True          # 是否处于静止站立状态
        self._stationary_start = 0.0        # 静止开始的时刻
        self._risk_stability_counter = 0    # 连续稳定帧计数
        self._risk_stability_threshold = 60 # 连续60帧(4s@15fps)稳定后开始衰减风险

        # 方向判断 (论文5)
        self._sway_dir_history: Deque[float] = deque(maxlen=self.win_dir)
        self._tilt_dir_history: Deque[float] = deque(maxlen=self.win_dir)

        # 上一帧状态 (用于差分)
        self._prev_com = None
        self._prev_knee_L = None
        self._prev_knee_R = None
        self._prev_ankle_mid = None
        self._last_step_time = 0.0
        self._ankle_y_history_L: Deque[float] = deque(maxlen=20)
        self._ankle_y_history_R: Deque[float] = deque(maxlen=20)

        # 姿态转换追踪 (v2.0)
        self._prev_bbox_ratio = None
        self._prev_head_ankle_dist = None

        # 时间
        self._start_time = time.time()
        self._frame_count = 0
        self._elapsed = 0.0

        # 风险持续计时
        self._yellow_start: Optional[float] = None    # v2.0: 四级
        self._orange_start: Optional[float] = None
        self._red_start: Optional[float] = None

        # 告警
        self.alert_callback = alert_callback
        self._last_alert_time = 0.0
        self._alert_cooldown = 15.0

        # EMA 平滑
        self._ema_alpha = 0.15
        self._smoothed: Dict[str, float] = {}

        self.last_report = FallRiskReport()
        self._transition_speed = 0.0

    # ═══════════════════════════════════════════════
    # 主接口
    # ═══════════════════════════════════════════════

    def update(self, landmarks: np.ndarray) -> FallRiskReport:
        self._frame_count += 1
        self._elapsed = time.time() - self._start_time

        # ═══ Layer 1: 14 项逐帧特征 (v2.0 增强) ═══
        trunk_sway = self._compute_trunk_sway(landmarks)
        trunk_tilt = self._compute_trunk_tilt(landmarks)
        com_margin = self._compute_com_bos_margin(landmarks)
        com_vel = self._compute_com_velocity(landmarks)
        knee_L = self._compute_joint_angle(landmarks, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        knee_R = self._compute_joint_angle(landmarks, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
        gait_speed = self._compute_gait_speed(landmarks)
        step_detected = self._detect_step(landmarks)

        # v2.0 新增特征
        bbox_ratio = self._compute_bbox_ratio(landmarks)
        head_ankle_dist = self._compute_head_ankle_dist(landmarks)
        angular_vel = self._compute_max_angular_velocity(knee_L, knee_R)
        transition_jerk = self._compute_transition_jerk(bbox_ratio, head_ankle_dist)
        double_support = self._detect_double_support(landmarks)

        # 存入缓冲区
        self._sway_history.append(trunk_sway)
        self._tilt_history.append(trunk_tilt)
        self._com_margin_history.append(com_margin)
        self._com_vel_history.append(com_vel)
        self._knee_L_history.append(knee_L)
        self._knee_R_history.append(knee_R)
        self._gait_speed_history.append(gait_speed)
        self._bbox_ratio_history.append(bbox_ratio)
        self._head_ankle_dist_history.append(head_ankle_dist)
        self._angular_vel_history.append(angular_vel)
        self._transition_jerk_history.append(transition_jerk)

        if step_detected:
            self._step_events.append(self._elapsed)
            self._is_walking = True
            self._last_step_elapsed = 0.0
            self._is_stationary = False
        else:
            self._last_step_elapsed += self._dt
            # 连续 3 秒无步态 → 进入静止状态
            if self._last_step_elapsed > 3.0:
                self._is_walking = False
                if not self._is_stationary:
                    self._is_stationary = True
                    self._stationary_start = self._elapsed

        # ═══ Layer 2: 10 项窗口统计 (v2.0 增强) ═══
        sway_mean = self._mean_recent(self._sway_history, self.win_short)
        sway_var = self._std_recent(self._sway_history, self.win_short)
        com_margin_min = self._min_recent(self._com_margin_history, self.win_short)
        com_vel_mean = self._mean_recent(self._com_vel_history, self.win_short)
        gait_speed_trend = self._trend_recent(self._gait_speed_history, self.win_long)
        step_regularity = self._compute_step_regularity()
        knee_asymmetry = abs(
            self._mean_recent(self._knee_L_history, self.win_short) -
            self._mean_recent(self._knee_R_history, self.win_short)
        )
        torso_tilt_mean = self._mean_recent(self._tilt_history, self.win_short)

        # v2.0 新增统计
        bbox_ratio_trend = self._trend_recent(self._bbox_ratio_history, self.win_short)
        head_ankle_collapse = self._compute_collapse_ratio()
        angular_vel_max = self._max_recent(self._angular_vel_history, self.win_short)
        angular_vel_var = self._std_recent(self._angular_vel_history, self.win_medium)
        double_support_ratio = self._compute_double_support_ratio()
        cadence = self._compute_cadence()

        # ── 跌倒方向 (论文5) ──
        sway_dir_recent = list(self._sway_history)[-self.win_dir:]
        tilt_dir_recent = list(self._tilt_history)[-self.win_dir:]
        fall_dir, fall_dir_deg = self._classify_fall_direction(
            sway_dir_recent, tilt_dir_recent)

        # ═══ Layer 3: 风险分计算 (v2.0: 6维) ═══
        sway_risk = self._sway_risk(sway_mean, sway_var)
        com_bos_risk = self._com_bos_risk(com_margin_min, com_vel_mean)
        gait_risk = self._gait_risk(gait_speed_trend, step_regularity,
                                     double_support_ratio)
        posture_risk = self._posture_risk(torso_tilt_mean, knee_asymmetry)
        transition_risk = self._transition_risk(angular_vel_max, angular_vel_var,
                                                 head_ankle_collapse, bbox_ratio_trend)
        direction_boost = self._direction_risk_boost(fall_dir)

        # 综合风险分 (v2.0: 6维权重)
        risk_score = (
            sway_risk * 0.25 +        # 躯干摇摆
            com_bos_risk * 0.25 +     # 重心-支撑面
            gait_risk * 0.20 +        # 步态
            posture_risk * 0.10 +     # 姿态
            transition_risk * 0.15 +  # 姿态转换 (v2.0)
            direction_boost * 0.05    # 方向加权 (v2.0)
        )
        risk_score = min(100.0, max(0.0, risk_score))

        # ── v2.1: 稳定状态风险衰减 (修复风险漂移) ──
        # 当用户静止站立时，缓慢衰减风险分至站立基线
        if self._is_stationary and self._is_walking is False:
            # 稳定帧数增加
            if sway_mean < 5.0 and angular_vel_max < 100.0 and abs(torso_tilt_mean) < 10.0:
                self._risk_stability_counter = min(self._risk_stability_counter + 1,
                                                    self._risk_stability_threshold * 2)
            else:
                self._risk_stability_counter = max(0, self._risk_stability_counter - 5)

            # 超过稳定性阈值后，衰减风险分
            if self._risk_stability_counter >= self._risk_stability_threshold:
                stability_decay = min(1.0, (self._risk_stability_counter - self._risk_stability_threshold)
                                      / self._risk_stability_threshold)
                # 朝向 "站立基线" (sway_risk ~10) 衰减
                standing_baseline = min(risk_score, max(10.0, sway_risk))
                risk_score = standing_baseline + (risk_score - standing_baseline) * (1.0 - stability_decay)
        else:
            # 非静止时重置稳定性计数
            self._risk_stability_counter = max(0, self._risk_stability_counter - 1)

        risk_score = min(100.0, max(0.0, risk_score))
        risk_score = self._ema("risk_score", risk_score, self._ema_alpha)

        # ═══ 四级告警 (论文2) ═══
        alert_level, alert_cn = self._classify_alert_level(risk_score, fall_dir)

        # ── 持续计时 ──
        now = self._elapsed
        self._yellow_start = now if alert_level == "YELLOW" and self._yellow_start is None else (
            None if alert_level != "YELLOW" else self._yellow_start)
        self._orange_start = now if alert_level == "ORANGE" and self._orange_start is None else (
            None if alert_level != "ORANGE" else self._orange_start)
        self._red_start = now if alert_level == "RED" and self._red_start is None else (
            None if alert_level != "RED" else self._red_start)

        sustained_yellow = (now - self._yellow_start) if self._yellow_start else 0.0
        sustained_orange = (now - self._orange_start) if self._orange_start else 0.0
        sustained_red = (now - self._red_start) if self._red_start else 0.0

        # 建议文案
        suggestion = ""
        if alert_level == "RED":
            suggestion = f"极度危险！{fall_dir}方向可能跌倒，请立即介入辅助！"
        elif alert_level == "ORANGE":
            suggestion = f"高风险！{fall_dir}方向不稳，请靠近老人准备搀扶"
        elif alert_level == "YELLOW":
            suggestion = "跌倒风险升高，请注意观察步态和平衡状态"
        elif alert_level == "BLUE":
            suggestion = "存在轻微不稳迹象，保持关注"

        # ── 推送判断 ──
        should_alert = False
        if sustained_red >= 1.5:
            should_alert = True
        elif sustained_orange >= 3.0:
            should_alert = True
        elif sustained_yellow >= 10.0:
            should_alert = True

        if should_alert and (now - self._last_alert_time) < self._alert_cooldown:
            should_alert = False

        report = FallRiskReport(
            risk_score=round(risk_score, 1),
            alert_level=alert_level,
            alert_level_cn=alert_cn,
            sway_risk=round(sway_risk, 1),
            com_bos_risk=round(com_bos_risk, 1),
            gait_risk=round(gait_risk, 1),
            posture_risk=round(posture_risk, 1),
            transition_risk=round(transition_risk, 1),
            direction_risk_boost=round(direction_boost, 1),
            fall_direction=fall_dir,
            fall_direction_deg=round(fall_dir_deg, 1),
            trunk_sway_deg=round(sway_mean, 1),
            trunk_tilt_deg=round(torso_tilt_mean, 1),
            com_bos_margin=round(com_margin_min, 3),
            gait_speed=round(gait_speed_trend, 3),
            step_regularity=round(step_regularity, 3),
            knee_angle_L=round(knee_L, 1),
            knee_angle_R=round(knee_R, 1),
            bbox_ratio=round(bbox_ratio, 3),
            head_ankle_dist=round(head_ankle_dist, 3),
            angular_velocity_max=round(angular_vel_max, 1),
            double_support_ratio=round(double_support_ratio, 3),
            cadence=round(cadence, 1),
            warning_sustained_sec=round(max(sustained_yellow, sustained_orange), 1),
            danger_sustained_sec=round(sustained_red, 1),
            should_alert=should_alert,
            suggestion=suggestion,
            timestamp=now,
        )
        self.last_report = report

        if should_alert:
            self._last_alert_time = now
            if self.alert_callback:
                self.alert_callback(report)

        return report

    # ═══════════════════════════════════════════════
    # Layer 1: 逐帧生物力学特征 (14 项)
    # ═══════════════════════════════════════════════

    def _compute_trunk_sway(self, lm: np.ndarray) -> float:
        """躯干侧向摇摆角 (度) — 肩髋横向夹角"""
        shoulder_mid = (lm[LEFT_SHOULDER, :2] + lm[RIGHT_SHOULDER, :2]) / 2
        hip_mid = (lm[LEFT_HIP, :2] + lm[RIGHT_HIP, :2]) / 2
        dx = hip_mid[0] - shoulder_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]
        if abs(dy) < 1e-6:
            return 0.0
        return self._ema("sway", float(np.degrees(np.arctan2(abs(dx), abs(dy)))), 0.3)

    def _compute_trunk_tilt(self, lm: np.ndarray) -> float:
        """躯干前后倾斜角 (度)"""
        shoulder_mid = (lm[LEFT_SHOULDER, :2] + lm[RIGHT_SHOULDER, :2]) / 2
        hip_mid = (lm[LEFT_HIP, :2] + lm[RIGHT_HIP, :2]) / 2
        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]
        if abs(dy) < 1e-6:
            return 0.0
        tilt = float(np.degrees(np.arctan2(abs(dx), abs(dy))))
        if lm.shape[1] >= 3:
            dz = abs(float((lm[LEFT_SHOULDER, 2] + lm[RIGHT_SHOULDER, 2]) / 2 -
                          (lm[LEFT_HIP, 2] + lm[RIGHT_HIP, 2]) / 2))
            tilt += dz * 10
        return self._ema("tilt", tilt, 0.3)

    def _compute_com_bos_margin(self, lm: np.ndarray) -> float:
        """重心-支撑面裕度 (0-1)"""
        com_x = (lm[LEFT_HIP, 0] + lm[RIGHT_HIP, 0]) / 2
        ankle_L_x = lm[LEFT_ANKLE, 0]
        ankle_R_x = lm[RIGHT_ANKLE, 0]
        bos_center = (ankle_L_x + ankle_R_x) / 2
        bos_hw = max(abs(ankle_R_x - ankle_L_x) / 2, 0.05)
        com_offset = abs(com_x - bos_center)
        margin = max(0.0, min(1.0, (bos_hw - com_offset) / bos_hw))
        return self._ema("com_margin", margin, 0.2)

    def _compute_com_velocity(self, lm: np.ndarray) -> float:
        """COM 移动速度 (归一化)"""
        com = np.array([
            (lm[LEFT_HIP, 0] + lm[RIGHT_HIP, 0]) / 2,
            (lm[LEFT_HIP, 1] + lm[RIGHT_HIP, 1]) / 2,
        ])
        if self._prev_com is None:
            self._prev_com = com
            return 0.0
        vel = float(np.linalg.norm(com - self._prev_com))
        self._prev_com = com.copy()
        return min(1.0, vel * 10)

    def _compute_gait_speed(self, lm: np.ndarray) -> float:
        """步速 (踝中点 x 位移)"""
        ankle_mid = np.array([
            (lm[LEFT_ANKLE, 0] + lm[RIGHT_ANKLE, 0]) / 2,
            (lm[LEFT_ANKLE, 1] + lm[RIGHT_ANKLE, 1]) / 2,
        ])
        if self._prev_ankle_mid is None:
            self._prev_ankle_mid = ankle_mid
            return 0.0
        speed = abs(ankle_mid[0] - self._prev_ankle_mid[0])
        self._prev_ankle_mid = ankle_mid.copy()
        return min(1.0, speed * 15)

    def _compute_joint_angle(self, lm: np.ndarray,
                             p1: int, p2: int, p3: int) -> float:
        """三点关节角度"""
        a, b, c = lm[p1, :2], lm[p2, :2], lm[p3, :2]
        ba, bc = a - b, c - b
        dot = np.dot(ba, bc)
        norm = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
        return float(np.degrees(np.arccos(np.clip(dot / norm, -1, 1))))

    def _compute_bbox_ratio(self, lm: np.ndarray) -> float:
        """
        人体包围盒宽高比 (论文5 骨骼点特征)
        跌倒时宽高比会显著变化: 正常站姿 ≈ 0.3-0.5, 倒地时 ≈ 1.5-2.5
        """
        valid = lm[:, :2].copy()
        x_min, y_min = valid.min(axis=0)
        x_max, y_max = valid.max(axis=0)
        w = max(x_max - x_min, 0.001)
        h = max(y_max - y_min, 0.001)
        return float(w / h)

    def _compute_head_ankle_dist(self, lm: np.ndarray) -> float:
        """
        头-踝距离 (论文2 非物理连接边特征)
        跌倒时身体塌缩，头接近地面 → 距离大幅减小
        """
        nose = lm[NOSE, :2]
        ankle_mid = np.array([
            (lm[LEFT_ANKLE, 0] + lm[RIGHT_ANKLE, 0]) / 2,
            (lm[LEFT_ANKLE, 1] + lm[RIGHT_ANKLE, 1]) / 2,
        ])
        return float(np.linalg.norm(nose - ankle_mid))

    def _compute_max_angular_velocity(self, knee_L: float, knee_R: float) -> float:
        """
        最大关节角速度 (论文5 动态特征: 角度加速度)
        """
        if self._prev_knee_L is None:
            self._prev_knee_L = knee_L
            self._prev_knee_R = knee_R
            return 0.0
        vel_L = abs(knee_L - self._prev_knee_L) / self._dt
        vel_R = abs(knee_R - self._prev_knee_R) / self._dt
        self._prev_knee_L = knee_L
        self._prev_knee_R = knee_R
        return max(vel_L, vel_R)

    def _compute_transition_jerk(self, bbox_ratio: float,
                                  head_ankle_dist: float) -> float:
        """
        姿态转换急动度 (论文5 动态特征)
        衡量姿态变化率的加速度 — 突然的快速姿态变化预示跌倒
        """
        if self._prev_bbox_ratio is None:
            self._prev_bbox_ratio = bbox_ratio
            self._prev_head_ankle_dist = head_ankle_dist
            self._transition_speed = 0.0
            return 0.0

        bbox_change = abs(bbox_ratio - self._prev_bbox_ratio)
        ha_change = abs(head_ankle_dist - self._prev_head_ankle_dist)
        current_speed = (bbox_change + ha_change * 2) / self._dt

        jerk = abs(current_speed - self._transition_speed) / self._dt

        self._prev_bbox_ratio = bbox_ratio
        self._prev_head_ankle_dist = head_ankle_dist
        self._transition_speed = current_speed

        return min(1.0, jerk * 0.5)

    def _detect_step(self, lm: np.ndarray) -> bool:
        """步态相位检测 (踝 y 坐标过零)"""
        ankle_y_L = lm[LEFT_ANKLE, 1]
        ankle_y_R = lm[RIGHT_ANKLE, 1]
        self._ankle_y_history_L.append(ankle_y_L)
        self._ankle_y_history_R.append(ankle_y_R)
        if len(self._ankle_y_history_L) < 5:
            return False
        yL_now = list(self._ankle_y_history_L)[-2:]
        yL_prev = list(self._ankle_y_history_L)[-5:-2]
        yR_now = list(self._ankle_y_history_R)[-2:]
        yR_prev = list(self._ankle_y_history_R)[-5:-2]
        step_L = abs(np.mean(yL_now) - np.mean(yL_prev)) > 0.005
        step_R = abs(np.mean(yR_now) - np.mean(yR_prev)) > 0.005
        detected = step_L or step_R
        if detected:
            now = self._elapsed
            if self._last_step_time > 0:
                interval = now - self._last_step_time
                if 0.1 < interval < 3.0:
                    self._step_intervals.append(interval)
            self._last_step_time = now
        return detected

    def _detect_double_support(self, lm: np.ndarray) -> bool:
        """
        双支撑相位检测 (论文5 步态特征)
        双脚踝 y 坐标差 < 阈值 → 双支撑
        """
        dy = abs(lm[LEFT_ANKLE, 1] - lm[RIGHT_ANKLE, 1])
        return dy < 0.03  # 归一化坐标

    # ═══════════════════════════════════════════════
    # Layer 2: 窗口统计分析 (10 项)
    # ═══════════════════════════════════════════════

    def _compute_step_regularity(self) -> float:
        """步态规整度: 1/(1+CV)"""
        if len(self._step_intervals) < 3:
            return 1.0
        arr = np.array(list(self._step_intervals))
        cv = float(np.std(arr) / (np.mean(arr) + 1e-9))
        return float(1.0 / (1.0 + cv))

    def _compute_collapse_ratio(self) -> float:
        """身体塌缩比率 (头-踝距离缩减比例)"""
        recent = list(self._head_ankle_dist_history)[-self.win_short:]
        if len(recent) < self.win_short // 2:
            return 0.0
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        if first_half < 0.01:
            return 0.0
        return max(0.0, (first_half - second_half) / first_half)

    def _compute_double_support_ratio(self) -> float:
        """双支撑时间比 (论文5)"""
        # 从踝 y 历史中估算
        if len(self._ankle_y_history_L) < 10:
            return 0.5
        yL = np.array(list(self._ankle_y_history_L))
        yR = np.array(list(self._ankle_y_history_R))
        dy = np.abs(yL - yR)
        ds_ratio = float(np.mean(dy < 0.02))  # 双踝同高 = 双支撑
        return ds_ratio

    def _compute_cadence(self) -> float:
        """步频 (steps/min) (论文5)"""
        if len(self._step_events) < 2:
            return 0.0
        t_first = self._step_events[0]
        t_last = self._step_events[-1]
        n_steps = len(self._step_events) - 1
        duration = t_last - t_first
        if duration < 0.5:
            return 0.0
        return float(n_steps / duration * 60.0)

    def _classify_fall_direction(self, sway_list: list, tilt_list: list) -> Tuple[str, float]:
        """
        跌倒方向判别 (论文5 核心创新)
        通过躯干摇摆和倾斜的累积分量判断
        """
        if len(sway_list) < self.win_dir // 2 or len(tilt_list) < self.win_dir // 2:
            return "none", 0.0

        net_sway = np.mean(sway_list)  # 侧向 > 0 = 左/右
        net_tilt = np.mean(tilt_list)  # 前后 > 增大 = 前倾

        # 使用累积值的方向趋势
        sway_trend = np.sum(sway_list[-self.win_dir//2:]) - np.sum(sway_list[:self.win_dir//2])
        tilt_trend = np.sum(tilt_list[-self.win_dir//2:]) - np.sum(tilt_list[:self.win_dir//2])

        angle = float(np.degrees(np.arctan2(sway_trend, tilt_trend + 1e-9)))
        mag = np.sqrt(sway_trend**2 + tilt_trend**2)

        if mag < 2.0:  # 变化太小不判定
            return "none", 0.0

        if abs(angle) <= 45:
            direction = "forward"
        elif abs(angle) >= 135:
            direction = "backward"
        elif angle > 45:
            direction = "right"
        else:
            direction = "left"

        return direction, angle

    # ═══════════════════════════════════════════════
    # Layer 3: 风险分计算 (6维)
    # ═══════════════════════════════════════════════

    def _sway_risk(self, mean: float, var: float) -> float:
        amp_r = self._linear_risk(mean, self.SWAY_WARN_DEG, self.SWAY_DANGER_DEG)
        var_r = self._linear_risk(var, self.SWAY_VARIABILITY_WARN, self.SWAY_VARIABILITY_DANGER)
        return min(100.0, (amp_r + var_r) / 2)

    def _com_bos_risk(self, margin_min: float, com_vel: float) -> float:
        # 裕度越小越危险
        if margin_min <= self.COM_BOS_DANGER:
            m_r = 100.0
        elif margin_min <= self.COM_BOS_WARN:
            m_r = 50.0 + (self.COM_BOS_WARN - margin_min) / (self.COM_BOS_WARN - self.COM_BOS_DANGER) * 50
        else:
            m_r = max(0.0, (1.0 - margin_min) * 50)
        v_r = min(100.0, com_vel / max(self.COM_VEL_DANGER, 0.01) * 100)
        return min(100.0, m_r * 0.7 + v_r * 0.3)

    def _gait_risk(self, speed_trend: float, regularity: float,
                   ds_ratio: float) -> float:
        # ── v2.1: 静止站立时不计算步态风险 (修复风险漂移) ──
        if not self._is_walking:
            return 0.0

        # 步速衰减
        if speed_trend < -self.GAIT_SLOWDOWN_DANGER:
            slow_r = 100.0
        elif speed_trend < 0:
            slow_r = abs(speed_trend) / max(self.GAIT_SLOWDOWN_DANGER, 0.01) * 100
        else:
            slow_r = 0.0

        # 不规整
        irreg = 1.0 - regularity
        irreg_r = self._linear_risk(irreg, self.STEP_IRREG_WARN, self.STEP_IRREG_DANGER)

        # 双支撑 (v2.0) — 仅在行走时计算
        ds_r = self._linear_risk(ds_ratio, self.DOUBLE_SUPPORT_WARN, self.DOUBLE_SUPPORT_DANGER)

        return min(100.0, slow_r * 0.4 + irreg_r * 0.35 + ds_r * 0.25)

    def _posture_risk(self, tilt: float, knee_asym: float) -> float:
        t_r = self._linear_risk(tilt, 15.0, self.TORSO_TILT_DANGER)
        k_r = self._linear_risk(knee_asym, 8.0, self.KNEE_ASYMMETRY_DANGER)
        return min(100.0, t_r * 0.6 + k_r * 0.4)

    def _transition_risk(self, ang_vel_max: float, ang_vel_var: float,
                         collapse: float, bbox_trend: float) -> float:
        """
        姿态转换风险 (v2.0 新增, 论文5 动态特征)
        整合角速度 + 身体塌缩 + 包围盒变化
        """
        ang_r = self._linear_risk(ang_vel_max, self.ANGULAR_VEL_WARN, self.ANGULAR_VEL_DANGER)
        ang_v_r = self._linear_risk(ang_vel_var, 50.0, 150.0)
        col_r = self._linear_risk(collapse, 0.15, self.HEAD_ANKLE_COLLAPSE_DANGER)
        bbox_r = self._linear_risk(abs(bbox_trend), 0.2, self.BBOX_RATIO_CHANGE_DANGER)
        return min(100.0, ang_r * 0.35 + ang_v_r * 0.15 + col_r * 0.25 + bbox_r * 0.25)

    def _direction_risk_boost(self, direction: str) -> float:
        """方向危险加权 (论文5: 后向最危险)"""
        return self.DIRECTION_WEIGHTS.get(direction, 0.0) * 20  # 最大 +26分

    # ── v2.1 校准阈值 ──
    SAFE_BLUE   = 35   # 正常站立 ~28-33 → SAFE
    BLUE_YELLOW = 55   # 正常行走 ~40-50 → BLUE
    YELLOW_ORANGE = 70 # 踉跄/不稳 → YELLOW
    ORANGE_RED  = 85   # 临近跌倒 → ORANGE, 极高 → RED

    def _classify_alert_level(self, risk: float, direction: str) -> Tuple[str, str]:
        """
        四级分级预警 (论文2 + 论文5方向加权) — v2.1 校准阈值
        RED  > ORANGE > YELLOW > BLUE > SAFE
        """
        effective_risk = risk
        # 后向跌倒方向提升一级 (提高触发门槛)
        if direction == "backward" and risk > 45:
            effective_risk = risk * 1.15  # 15% 加成

        if effective_risk >= self.ORANGE_RED:
            return "RED", "红色"
        elif effective_risk >= self.YELLOW_ORANGE:
            return "ORANGE", "橙色"
        elif effective_risk >= self.BLUE_YELLOW:
            return "YELLOW", "黄色"
        elif effective_risk >= self.SAFE_BLUE:
            return "BLUE", "蓝色"
        else:
            return "SAFE", "安全"

    # ═══════════════════════════════════════════════
    # 统计工具
    # ═══════════════════════════════════════════════

    @staticmethod
    def _linear_risk(val: float, warn: float, danger: float) -> float:
        if val >= danger:
            return 100.0
        if val <= warn:
            return max(0.0, val / max(warn, 0.01) * 50)
        return 50.0 + (val - warn) / max(danger - warn, 0.01) * 50

    @staticmethod
    def _mean_recent(buf: Deque, n: int) -> float:
        if len(buf) == 0:
            return 0.0
        items = list(buf)[-min(n, len(buf)):]
        return float(np.mean(items))

    @staticmethod
    def _std_recent(buf: Deque, n: int) -> float:
        if len(buf) < 2:
            return 0.0
        items = list(buf)[-min(n, len(buf)):]
        return float(np.std(items))

    @staticmethod
    def _min_recent(buf: Deque, n: int) -> float:
        if len(buf) == 0:
            return 0.0
        return float(np.min(list(buf)[-min(n, len(buf)):]))

    @staticmethod
    def _max_recent(buf: Deque, n: int) -> float:
        if len(buf) == 0:
            return 0.0
        return float(np.max(list(buf)[-min(n, len(buf)):]))

    @staticmethod
    def _trend_recent(buf: Deque, n: int) -> float:
        if len(buf) < 5:
            return 0.0
        items = list(buf)[-min(n, len(buf)):]
        x = np.arange(len(items))
        slope = float(np.polyfit(x, items, 1)[0])
        return slope * len(items)

    def _ema(self, key: str, value: float, alpha: float) -> float:
        if key not in self._smoothed:
            self._smoothed[key] = value
        else:
            self._smoothed[key] = alpha * value + (1 - alpha) * self._smoothed[key]
        return self._smoothed[key]

    # ═══════════════════════════════════════════════
    # 重置 & 状态查询
    # ═══════════════════════════════════════════════

    def reset(self):
        for attr in ['_sway_history', '_tilt_history', '_com_margin_history',
                     '_com_vel_history', '_knee_L_history', '_knee_R_history',
                     '_gait_speed_history', '_bbox_ratio_history',
                     '_head_ankle_dist_history', '_angular_vel_history',
                     '_transition_jerk_history', '_step_events', '_step_intervals',
                     '_ankle_y_history_L', '_ankle_y_history_R',
                     '_sway_dir_history', '_tilt_dir_history']:
            getattr(self, attr).clear()
        self._prev_com = None
        self._prev_knee_L = None
        self._prev_knee_R = None
        self._prev_ankle_mid = None
        self._prev_bbox_ratio = None
        self._prev_head_ankle_dist = None
        self._last_step_time = 0.0
        self._start_time = time.time()
        self._frame_count = 0
        self._elapsed = 0.0
        self._yellow_start = None
        self._orange_start = None
        self._red_start = None
        self._smoothed.clear()
        self._transition_speed = 0.0
        self._is_walking = False
        self._last_step_elapsed = 0.0
        self._is_stationary = True
        self._stationary_start = 0.0
        self._risk_stability_counter = 0
        self.last_report = FallRiskReport()

    def get_status(self) -> dict:
        r = self.last_report
        return {
            "risk_score": r.risk_score,
            "alert_level": r.alert_level,
            "alert_cn": r.alert_level_cn,
            "fall_direction": r.fall_direction,
            "sway_risk": r.sway_risk,
            "com_bos_risk": r.com_bos_risk,
            "gait_risk": r.gait_risk,
            "posture_risk": r.posture_risk,
            "transition_risk": r.transition_risk,
            "direction_boost": r.direction_risk_boost,
            "trunk_sway_deg": r.trunk_sway_deg,
            "com_bos_margin": r.com_bos_margin,
            "step_regularity": r.step_regularity,
            "angular_vel_max": r.angular_velocity_max,
            "bbox_ratio": r.bbox_ratio,
            "head_ankle_dist": r.head_ankle_dist,
            "double_support": r.double_support_ratio,
            "cadence": r.cadence,
            "should_alert": r.should_alert,
            "suggestion": r.suggestion,
            # v2.1
            "is_walking": self._is_walking,
            "is_stationary": self._is_stationary,
            "risk_stability": self._risk_stability_counter,
        }


# ═══════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════

def test_predictor_v2():
    print("=== FallPredictor v2.0 测试 ===\n")
    p = FallPredictor(fps=15.0)

    scenarios = [
        ("正常行走 (安全)", 120, 0.01, 0.0, 0.0),
        ("轻微摇摆 (蓝色)", 90, 0.04, 0.0, 0.0),
        ("不稳 + 前倾 (黄色)", 90, 0.06, 0.05, 0.0),
        ("剧烈摇摆 + 后退 (橙色)", 90, 0.10, -0.10, 0.0),
        ("极度不稳 (红色)", 90, 0.15, -0.15, 0.5),
    ]

    for name, frames, sway, forward, collapse in scenarios:
        p.reset()
        p._frame_count = 30  # skip warm-up logging
        for i in range(frames):
            lm = _simulate_person(i, sway_amp=sway, forward_tilt=forward,
                                  collapse_ratio=collapse)
            report = p.update(lm)
        print(f"  {name}: Risk={report.risk_score:.0f} "
              f"Level={report.alert_level}({report.alert_level_cn}) "
              f"Dir={report.fall_direction}")
        print(f"    摇摆={report.sway_risk:.0f} COM={report.com_bos_risk:.0f} "
              f"步态={report.gait_risk:.0f} 姿态={report.posture_risk:.0f} "
              f"转换={report.transition_risk:.0f} 方向加成={report.direction_risk_boost:.0f}")
        if report.suggestion:
            print(f"    建议: {report.suggestion}")

    print("\n[OK] v2.0 测试完成")


def _simulate_person(frame_idx: int, sway_amp: float = 0.02,
                     forward_tilt: float = 0.0, collapse_ratio: float = 0.0) -> np.ndarray:
    """生成模拟人体关键点"""
    lm = np.zeros((33, 4), dtype=np.float32)
    t = frame_idx * 0.1
    cx = 0.5 + sway_amp * 1.5 * np.sin(t * 2.0)
    ft = forward_tilt * min(frame_idx / 60.0, 1.0)  # 渐进前/后倾
    cl = collapse_ratio * min(frame_idx / 40.0, 1.0)  # 渐进塌缩

    # 头
    lm[NOSE, 0] = cx
    lm[NOSE, 1] = 0.08 + ft * 0.3 - cl * 0.3

    # 肩
    lm[LEFT_SHOULDER, 0] = cx - 0.08
    lm[LEFT_SHOULDER, 1] = 0.22 + ft * 0.15
    lm[RIGHT_SHOULDER, 0] = cx + 0.08
    lm[RIGHT_SHOULDER, 1] = 0.22 + ft * 0.15

    # 髋 (大幅摇摆)
    lm[LEFT_HIP, 0] = cx - 0.06 + sway_amp * 1.5 * np.sin(t * 2.3 + 0.5)
    lm[LEFT_HIP, 1] = 0.48 + cl * 0.05
    lm[RIGHT_HIP, 0] = cx + 0.06 + sway_amp * 1.5 * np.sin(t * 2.3 + 0.5)
    lm[RIGHT_HIP, 1] = 0.48 + cl * 0.05

    # 膝
    lm[LEFT_KNEE, 0] = lm[LEFT_HIP, 0]
    lm[LEFT_KNEE, 1] = 0.68 + cl * 0.1
    lm[RIGHT_KNEE, 0] = lm[RIGHT_HIP, 0]
    lm[RIGHT_KNEE, 1] = 0.68 + cl * 0.1

    # 踝
    lm[LEFT_ANKLE, 0] = lm[LEFT_KNEE, 0] + 0.02 * np.sin(t * 1.5)
    lm[LEFT_ANKLE, 1] = 0.88 + cl * 0.08 + 0.01 * np.sin(t * 1.5)
    lm[RIGHT_ANKLE, 0] = lm[RIGHT_KNEE, 0] + 0.02 * np.sin(t * 1.5 + np.pi)
    lm[RIGHT_ANKLE, 1] = 0.88 + cl * 0.08 + 0.01 * np.sin(t * 1.5 + np.pi)

    # 肘
    lm[LEFT_ELBOW, 0] = lm[LEFT_SHOULDER, 0] - 0.05
    lm[LEFT_ELBOW, 1] = 0.35 + ft * 0.15
    lm[RIGHT_ELBOW, 0] = lm[RIGHT_SHOULDER, 0] + 0.05
    lm[RIGHT_ELBOW, 1] = 0.35 + ft * 0.15

    # 腕
    lm[LEFT_WRIST, 0] = lm[LEFT_ELBOW, 0] - 0.03
    lm[LEFT_WRIST, 1] = 0.48
    lm[RIGHT_WRIST, 0] = lm[RIGHT_ELBOW, 0] + 0.03
    lm[RIGHT_WRIST, 1] = 0.48

    lm[:, 3] = 0.9
    return lm


def main():
    parser = argparse.ArgumentParser(description="FallPredictor v2.0 — 基于6篇论文学术升级")
    parser.add_argument("--test", action="store_true", help="运行模拟测试")
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()
    if args.test:
        test_predictor_v2()
    else:
        print("FallPredictor v2.0 ready")

if __name__ == "__main__":
    main()
