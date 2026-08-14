#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
medication_reminder.py — 服药提醒模块

定时提醒 + 可选姿态验证（确认老人在摄像头前）。
"""

import time
import json
import os
from typing import List, Dict, Optional, Tuple
import numpy as np

# MediaPipe 关键点索引
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


class MedicationReminder:
    """
    服药提醒器
    - 支持多个定时提醒
    - 提醒时可选姿态验证（检测老人在摄像头前）
    - 状态持久化到 JSON
    """

    def __init__(self, state_path: str = None):
        """
        Args:
            state_path: 状态文件路径 (None 则用默认路径)
        """
        if state_path is None:
            state_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "data", "medication_state.json"
            )
        self.state_path = os.path.abspath(state_path)
        self.schedules: List[Dict] = []  # 提醒时间表
        self.reminded_today: Dict[str, bool] = {}  # 今日已提醒记录
        self.person_present = False
        self.last_alert_time = 0.0
        self.alert_cooldown_s = 300  # 提醒冷却(秒)

        # 加载状态
        self._load_state()

    def add_schedule(self, hour: int, minute: int, name: str = ""):
        """添加一个服药提醒时间"""
        self.schedules.append({
            "hour": hour,
            "minute": minute,
            "name": name or f"{hour:02d}:{minute:02d}",
        })
        self.schedules.sort(key=lambda s: (s["hour"], s["minute"]))
        self._save_state()
        return len(self.schedules)

    def remove_schedule(self, index: int):
        if 0 <= index < len(self.schedules):
            removed = self.schedules.pop(index)
            self._save_state()
            return removed
        return None

    def should_remind(self, current_hour: int, current_minute: int) -> bool:
        """检查当前时间是否需要提醒"""
        now = time.time()
        if now - self.last_alert_time < self.alert_cooldown_s:
            return False

        today = time.strftime("%Y-%m-%d", time.localtime())
        for sched in self.schedules:
            if sched["hour"] == current_hour and current_minute == sched["minute"]:
                key = f"{today}_{sched['name']}"
                if not self.reminded_today.get(key, False):
                    return True
        return False

    def mark_reminded(self, current_hour: int, current_minute: int) -> str:
        """标记已提醒, 返回提醒消息"""
        self.last_alert_time = time.time()
        today = time.strftime("%Y-%m-%d", time.localtime())

        for sched in self.schedules:
            if sched["hour"] == current_hour and current_minute == sched["minute"]:
                key = f"{today}_{sched['name']}"
                self.reminded_today[key] = True
                # 清理旧日期的记录
                self._cleanup_old_records(today)
                self._save_state()
                return f"💊 服药提醒: {sched['name']} — 请按时服药！"
        return ""

    def check_person_present(self, landmarks: np.ndarray) -> bool:
        """
        通过关键点检查是否有人在摄像头前
        - 鼻子 + 双肩可见性 > 0.5
        - 身体大小合理 (y 跨度 > 0.1)
        """
        if landmarks.shape[0] < 33:
            return False

        vis = landmarks[:, 3] if landmarks.shape[1] >= 4 else np.ones(33)
        nose_vis = vis[NOSE]
        shoulder_vis = min(vis[LEFT_SHOULDER], vis[RIGHT_SHOULDER])

        y = landmarks[:, 1]
        body_span = np.ptp(y[np.isfinite(y)])

        self.person_present = (
            nose_vis > 0.5 and shoulder_vis > 0.5 and body_span > 0.1
        )
        return self.person_present

    def update(self, landmarks: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        每帧调用: 检查是否需要提醒

        Args:
            landmarks: 可选, 用于验证老人在摄像头前

        Returns:
            (alert_triggered, alert_message)
        """
        now = time.localtime()
        current_hour, current_minute = now.tm_hour, now.tm_min

        if not self.should_remind(current_hour, current_minute):
            return False, ""

        # 验证是否有老人在
        if landmarks is not None:
            if not self.check_person_present(landmarks):
                return False, ""  # 没人在不提醒

        msg = self.mark_reminded(current_hour, current_minute)
        return True, msg

    def _load_state(self):
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.schedules = data.get("schedules", [])
                    self.reminded_today = data.get("reminded_today", {})
        except (json.JSONDecodeError, IOError):
            self.schedules = []
            self.reminded_today = {}

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump({
                "schedules": self.schedules,
                "reminded_today": self.reminded_today,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2, ensure_ascii=False)

    def _cleanup_old_records(self, today: str):
        self.reminded_today = {
            k: v for k, v in self.reminded_today.items()
            if k.startswith(today)
        }

    def get_status(self) -> dict:
        return {
            "schedules": self.schedules,
            "person_present": self.person_present,
            "reminded_today": len(self.reminded_today),
        }


# ============================================================
# 独立测试
# ============================================================

def test():
    print("=== 服药提醒器测试 ===\n")

    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), "test_medication.json")
    reminder = MedicationReminder(state_path=tmp)

    # 添加提醒
    reminder.add_schedule(8, 0, "早餐后")
    reminder.add_schedule(12, 30, "午餐后")
    reminder.add_schedule(18, 0, "晚餐后")
    print(f"  已添加 {len(reminder.schedules)} 个提醒时间")

    # 模拟当前是提醒时间
    now = time.localtime()
    print(f"  当前时间: {now.tm_hour:02d}:{now.tm_min:02d}")

    # 测试提醒触发
    alerted, msg = reminder.update()
    print(f"  提醒: {alerted}, 消息: '{msg}'")

    if alerted:
        # 测试标记后不重复
        alerted2, _ = reminder.update()
        print(f"  二次检查: {alerted2} (应为 False)")

    # 测试姿态验证
    dummy_lm = np.zeros((33, 4))
    dummy_lm[:, 3] = 0.9
    dummy_lm[NOSE, :2] = [0.5, 0.2]
    dummy_lm[LEFT_SHOULDER, :2] = [0.4, 0.3]
    dummy_lm[RIGHT_SHOULDER, :2] = [0.6, 0.3]
    present = reminder.check_person_present(dummy_lm)
    print(f"  人员检测: {'有人' if present else '无人'}")

    # 清理
    if os.path.exists(tmp):
        os.remove(tmp)
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test()
