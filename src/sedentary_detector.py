#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sedentary_detector.py — 久坐检测模块

基于关键点判断坐姿状态, 累计连续坐姿时长, 超时触发告警。
"""

import time
import numpy as np
from typing import Tuple

# MediaPipe 关键点索引
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
NOSE = 0


class SedentaryDetector:
    """
    久坐检测器
    - 基于关键点判断坐姿: 髋部低于肩部一定比例, 膝部接近髋部高度
    - 连续坐姿超时触发提醒
    - 置信度加权防止误判
    """

    def __init__(self,
                 alert_minutes: float = 60.0,       # 触发告警的久坐时长(分钟)
                 warn_minutes: float = 45.0,         # 预警提醒时长
                 min_sitting_ratio: float = 0.6,     # 连续坐姿占比阈值
                 check_interval_s: float = 30.0):    # 检查间隔(秒)
        self.alert_minutes = alert_minutes
        self.warn_minutes = warn_minutes
        self.min_ratio = min_sitting_ratio
        self.check_interval_s = check_interval_s

        # 状态
        self.sitting_start_time: float = 0.0  # 本次坐姿开始时间
        self.sitting_frames = 0
        self.total_frames = 0
        self.is_sitting = False
        self.last_check_time = time.time()

        # 告警状态
        self.warned = False      # 是否已发预警
        self.alerted = False     # 是否已发告警
        self.last_standing_time = time.time()  # 最后一次站立时间

    def update(self, landmarks: np.ndarray) -> Tuple[bool, str]:
        """
        每帧调用

        Args:
            landmarks: MediaPipe 33关键点 (33, 3) 或 (33, 4)

        Returns:
            (is_sitting, alert_message)
                is_sitting: 当前是否坐姿
                alert_message: 告警消息 (空字符串表示无告警)
        """
        if landmarks.shape[0] < 33:
            return False, ""

        y = landmarks[:, 1]  # y 坐标
        vis = landmarks[:, 3] if landmarks.shape[1] >= 4 else np.ones(33)

        # 计算关键关节点高度(归一化 y 坐标, 越小越靠上)
        hip_y = (y[LEFT_HIP] + y[RIGHT_HIP]) / 2
        knee_y = (y[LEFT_KNEE] + y[RIGHT_KNEE]) / 2
        shoulder_y = (y[LEFT_SHOULDER] + y[RIGHT_SHOULDER]) / 2

        # 可见性
        hip_vis = min(vis[LEFT_HIP], vis[RIGHT_HIP])
        knee_vis = min(vis[LEFT_KNEE], vis[RIGHT_KNEE])

        # 坐姿判断:
        # 1. 髋部明显低于肩部 (髋/肩 > 1.3 = 髋在肩下方足够多)
        # 2. 膝部接近髋部高度 (膝/髋 < 1.3 = 膝盖不在髋下方很远 = 非站立)
        torso_ratio = hip_y / (shoulder_y + 1e-6)
        leg_ratio = knee_y / (hip_y + 1e-6)

        is_sitting_now = (
            hip_vis > 0.3 and knee_vis > 0.3 and
            torso_ratio > 1.3 and leg_ratio < 1.3
        )

        self.total_frames += 1
        if is_sitting_now:
            self.sitting_frames += 1

        # 定期检查
        now = time.time()
        if now - self.last_check_time < self.check_interval_s:
            self.is_sitting = is_sitting_now
            return is_sitting_now, ""

        self.last_check_time = now
        sitting_ratio = self.sitting_frames / max(self.total_frames, 1)

        # 判断是否处于坐姿状态
        was_sitting = self.is_sitting
        self.is_sitting = sitting_ratio >= self.min_ratio

        # 状态转换
        if self.is_sitting and not was_sitting:
            # 开始坐姿
            self.sitting_start_time = now
            self.warned = False
            self.alerted = False
        elif not self.is_sitting and was_sitting:
            # 结束坐姿
            self.last_standing_time = now
            self.sitting_frames = 0
            self.total_frames = 0
            self.sitting_start_time = 0
            self.warned = False
            self.alerted = False

        # 检查久坐时长
        if self.is_sitting and self.sitting_start_time > 0:
            sitting_duration = now - self.sitting_start_time
            sitting_min = sitting_duration / 60.0

            if sitting_min >= self.alert_minutes and not self.alerted:
                self.alerted = True
                return True, f"⚠️ 已连续久坐 {sitting_min:.0f} 分钟，请起身活动！"
            elif sitting_min >= self.warn_minutes and not self.warned:
                self.warned = True
                return True, f"💡 已坐 {sitting_min:.0f} 分钟，建议起身活动"

        self.is_sitting = is_sitting_now
        return is_sitting_now, ""

    def reset(self):
        """重置状态"""
        self.sitting_start_time = 0
        self.sitting_frames = 0
        self.total_frames = 0
        self.is_sitting = False
        self.warned = False
        self.alerted = False
        self.last_check_time = time.time()

    def get_status(self) -> dict:
        """获取当前状态"""
        sitting_min = 0
        if self.is_sitting and self.sitting_start_time > 0:
            sitting_min = (time.time() - self.sitting_start_time) / 60.0
        return {
            "is_sitting": self.is_sitting,
            "sitting_minutes": round(sitting_min, 1),
            "alert_minutes": self.alert_minutes,
            "warned": self.warned,
            "alerted": self.alerted,
        }


# ============================================================
# 独立测试
# ============================================================

def test():
    """用模拟关键点测试久坐检测"""
    import random
    print("=== 久坐检测器测试 ===\n")

    detector = SedentaryDetector(alert_minutes=0.5, warn_minutes=0.3, check_interval_s=2.0)

    # 模拟站立
    def standing_lm():
        lm = np.zeros((33, 4))
        lm[:, 3] = 0.9
        lm[:, 1] = 0.5  # 站立时身体较高
        lm[NOSE, 1] = 0.1
        lm[LEFT_SHOULDER, 1] = 0.2
        lm[RIGHT_SHOULDER, 1] = 0.2
        lm[LEFT_HIP, 1] = 0.45
        lm[RIGHT_HIP, 1] = 0.45
        lm[LEFT_KNEE, 1] = 0.75
        lm[RIGHT_KNEE, 1] = 0.75
        lm[LEFT_ANKLE, 1] = 0.98
        lm[RIGHT_ANKLE, 1] = 0.98
        lm[:, 0] = 0.5
        return lm

    # 模拟坐姿
    def sitting_lm():
        lm = np.zeros((33, 4))
        lm[:, 3] = 0.9
        lm[:, 1] = 0.7  # 坐着时身体整体下移
        lm[NOSE, 1] = 0.3
        lm[LEFT_SHOULDER, 1] = 0.4
        lm[RIGHT_SHOULDER, 1] = 0.4
        lm[LEFT_HIP, 1] = 0.65
        lm[RIGHT_HIP, 1] = 0.65
        lm[LEFT_KNEE, 1] = 0.75
        lm[RIGHT_KNEE, 1] = 0.75
        lm[LEFT_ANKLE, 1] = 0.85
        lm[RIGHT_ANKLE, 1] = 0.85
        lm[:, 0] = 0.5
        return lm

    print("模拟站立 5秒...")
    for _ in range(5):
        sitting, msg = detector.update(standing_lm())
        print(f"  坐姿={sitting}, 消息='{msg}'")
        time.sleep(0.1)

    print("\n模拟久坐 3秒...")
    for _ in range(4):
        sitting, msg = detector.update(sitting_lm())
        status = detector.get_status()
        print(f"  坐姿={sitting}, 累计={status['sitting_minutes']}分钟, 消息='{msg}'")
        time.sleep(1.0)

    print(f"\n状态: {detector.get_status()}")
    print("=== 测试完成 ===")


if __name__ == "__main__":
    test()
