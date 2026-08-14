#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alert_manager.py — 告警通知通道 v1.0
=====================================
当 risk_score >= ORANGE 或 ML 判 Fall 时自动:
  1. 保存告警截图到 alerts/ 目录
  2. 播放本地音频警报
  3. 告警限流 (冷却时间避免连续重复)
  4. 告警历史记录

用法:
    mgr = AlertManager(alerts_dir=r"E:\老人跌倒\alerts")
    mgr.update(risk_score=65, alert_level="ORANGE",
               ml_fall_triggered=False, frame=frame_bgr)

设计原则:
  - 音频用 winsound (Windows 零依赖), pygame 可选增强
  - 截图命名: alert_S{severity}_F{frame}_{timestamp}.jpg
  - 冷却: ORANGE 30s, RED 15s, ML Fall 10s (同一类型)
  - 不同级别告警不互斥 (ORANGE + ML Fall 同时触发)
"""

import os
import sys
import time
import cv2
import platform
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

# ── 音频后端选择 ──
_WINSOUND_AVAILABLE = False
_PYGAME_AVAILABLE = False
_PYGAME_INITIALIZED = False

if platform.system() == "Windows":
    try:
        import winsound
        _WINSOUND_AVAILABLE = True
    except ImportError:
        pass

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    pass

# ── 数据类 ──

@dataclass
class AlertEvent:
    """一次告警事件"""
    alert_type: str          # "risk_orange" | "risk_red" | "ml_fall"
    risk_score: float
    alert_level: str
    ml_fall_triggered: bool
    fall_direction: str
    screenshot_path: str
    timestamp: float
    iso_time: str


class AlertManager:
    """告警通知管理器"""

    # 冷却时间 (秒) — v1.1: 增强冷却避免告警洪流
    GLOBAL_COOLDOWN = 20.0      # 任意两次告警的最小间隔
    COOLDOWN_ORANGE = 60.0      # ORANGE 级别告警冷却
    COOLDOWN_RED = 30.0         # RED 级别告警冷却
    COOLDOWN_ML_FALL = 20.0     # ML 摔倒告警冷却

    # 截图保存配置
    SCREENSHOT_QUALITY = 85      # JPEG 质量 (0-100)
    MAX_SCREENSHOTS = 200        # 告警目录最大截图数 (超出删最旧的)

    def __init__(self, alerts_dir: str = r"E:\老人跌倒\alerts",
                 enable_audio: bool = True,
                 enable_screenshots: bool = True,
                 alert_volume: float = 0.8):
        """
        Args:
            alerts_dir: 告警截图保存目录
            enable_audio: 是否播放音频
            enable_screenshots: 是否保存截图
            alert_volume: 音量 0.0-1.0 (仅 pygame 后端)
        """
        self.alerts_dir = alerts_dir
        self.enable_audio = enable_audio
        self.enable_screenshots = enable_screenshots
        self.alert_volume = max(0.0, min(1.0, alert_volume))

        os.makedirs(self.alerts_dir, exist_ok=True)

        # 冷却追踪: {alert_type: last_trigger_time}
        self._cooldowns: Dict[str, float] = {}
        # 全局冷却: 任意告警间的最小间隔
        self._last_global_alert: float = 0.0
        # 告警历史
        # 告警历史
        self.history: List[AlertEvent] = []
        # 音频资源路径
        self._sound_files: Dict[str, str] = {}
        self._pygame_sounds: Dict[str, object] = {}

        # 初始化音频
        if self.enable_audio:
            self._init_audio()

        # 清理旧截图
        self._cleanup_old_screenshots()

        if enable_screenshots:
            print(f"  [ALERT] 截图目录: {self.alerts_dir}")
        if enable_audio:
            backend = "pygame" if _PYGAME_AVAILABLE else ("winsound" if _WINSOUND_AVAILABLE else "none")
            print(f"  [ALERT] 音频后端: {backend}")

    # ════════════════════════════════════════════════════
    # 每帧调用
    # ════════════════════════════════════════════════════

    def update(self, risk_score: float = 0.0, alert_level: str = "SAFE",
               ml_fall_triggered: bool = False,
               fall_direction: str = "none",
               frame = None) -> Optional[AlertEvent]:
        """
        每帧调用。当满足告警条件时自动触发通知。

        Args:
            risk_score: 当前风险分 (0-100)
            alert_level: 当前告警级别 (SAFE/BLUE/YELLOW/ORANGE/RED)
            ml_fall_triggered: ML 分类器是否判定摔倒
            fall_direction: 跌倒方向
            frame: OpenCV BGR 帧图像 (用于截图)

        Returns:
            AlertEvent 如果触发了告警, 否则 None
        """
        now = time.time()

        # ── 判定哪些告警需要触发 ──
        triggers = []

        if alert_level == "RED":
            triggers.append(("risk_red", self.COOLDOWN_RED))
        elif alert_level == "ORANGE":
            triggers.append(("risk_orange", self.COOLDOWN_ORANGE))

        if ml_fall_triggered:
            triggers.append(("ml_fall", self.COOLDOWN_ML_FALL))

        if not triggers:
            return None

        # ── 全局冷却检查 (v1.1) ──
        if now - self._last_global_alert < self.GLOBAL_COOLDOWN:
            return None

        # ── 检查各类型冷却 ──
        triggered_any = False
        for alert_type, cooldown in triggers:
            last = self._cooldowns.get(alert_type, 0)
            if now - last >= cooldown:
                self._cooldowns[alert_type] = now
                triggered_any = True

        if not triggered_any:
            return None

        # 更新全局冷却时间
        self._last_global_alert = now

        # ── 执行通知 ──
        screenshot_path = ""
        from datetime import datetime
        iso_time = datetime.now().isoformat(timespec='seconds')

        # 截图
        if self.enable_screenshots and frame is not None:
            screenshot_path = self._save_screenshot(frame, alert_level,
                                                     ml_fall_triggered, iso_time)

        # 音频
        if self.enable_audio:
            self._play_alert(alert_level, ml_fall_triggered)

        # 记录
        event = AlertEvent(
            alert_type="+".join(t for t, _ in triggers),
            risk_score=risk_score,
            alert_level=alert_level,
            ml_fall_triggered=ml_fall_triggered,
            fall_direction=fall_direction,
            screenshot_path=screenshot_path,
            timestamp=now,
            iso_time=iso_time,
        )
        self.history.append(event)
        # 保留最近 500 条
        if len(self.history) > 500:
            self.history = self.history[-500:]

        return event

    # ════════════════════════════════════════════════════
    # 截图
    # ════════════════════════════════════════════════════

    def _save_screenshot(self, frame, alert_level: str,
                         ml_fall_triggered: bool,
                         iso_time: str) -> str:
        """保存告警截图, 返回路径"""
        # 文件名
        severity = alert_level.lower()
        if ml_fall_triggered:
            severity = f"{severity}_fall"
        ts = iso_time.replace(':', '-').replace('T', '_')
        fname = f"alert_{severity}_{ts}.jpg"
        fpath = os.path.join(self.alerts_dir, fname)

        # 在帧上叠加告警水印
        watermarked = frame.copy()
        h, w = watermarked.shape[:2]
        overlay = watermarked.copy()
        # 红色边框闪烁效果
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), 8)
        # 顶部告警横幅
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 180), -1)
        alert_text = f"ALERT: {alert_level}"
        if ml_fall_triggered:
            alert_text += " | FALL DETECTED"
        cv2.putText(overlay, alert_text, (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        watermarked = cv2.addWeighted(overlay, 0.7, watermarked, 0.3, 0)

        # 底部时间戳
        cv2.putText(watermarked, iso_time, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imwrite(fpath, watermarked, [cv2.IMWRITE_JPEG_QUALITY, self.SCREENSHOT_QUALITY])

        # 如果路径包含中文, cv2.imwrite 可能有编码问题 → 用 numpy + 二进制写入兜底
        if not os.path.exists(fpath):
            import numpy as np
            _, buf = cv2.imencode('.jpg', watermarked,
                                  [cv2.IMWRITE_JPEG_QUALITY, self.SCREENSHOT_QUALITY])
            with open(fpath, 'wb') as f:
                f.write(buf.tobytes())

        return fpath

    def _cleanup_old_screenshots(self):
        """如果截图超过 MAX_SCREENSHOTS, 删除最旧的"""
        try:
            files = [os.path.join(self.alerts_dir, f)
                     for f in os.listdir(self.alerts_dir)
                     if f.endswith('.jpg')]
            if len(files) <= self.MAX_SCREENSHOTS:
                return
            files.sort(key=os.path.getmtime)
            for f in files[:len(files) - self.MAX_SCREENSHOTS]:
                os.remove(f)
        except Exception:
            pass

    # ════════════════════════════════════════════════════
    # 音频
    # ════════════════════════════════════════════════════

    def _init_audio(self):
        """初始化音频后端"""
        if _PYGAME_AVAILABLE:
            try:
                if not _PYGAME_INITIALIZED:
                    pygame.mixer.init(frequency=44100, size=-16, channels=1)
                # 生成三种告警音
                self._generate_alert_sounds()
                print("  [ALERT] 告警音已生成 (pygame 合成)")
                return
            except Exception as e:
                print(f"  [ALERT] pygame 初始化失败: {e}, 回退 winsound")

        if _WINSOUND_AVAILABLE:
            print("  [ALERT] 使用系统蜂鸣 (winsound)")
        else:
            print("  [ALERT] 无可用音频后端, 告警静默")

    def _generate_alert_sounds(self):
        """用 pygame 合成告警音频并缓存"""
        import numpy as np

        sample_rate = 44100
        duration = 0.3  # 秒

        sounds = {}

        # ORANGE: 双音交替 (800Hz + 1000Hz)
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        half = len(t) // 2
        wave = np.zeros(len(t), dtype=np.float32)
        wave[:half] = np.sin(2 * np.pi * 800 * t[:half])
        wave[half:] = np.sin(2 * np.pi * 1000 * t[half:])
        # 衰减包络
        env = np.linspace(1.0, 0.1, len(t))
        wave *= env * self.alert_volume
        sounds['risk_orange'] = wave

        # RED: 快速脉冲 (1500Hz, 短脉冲 × 3)
        pulse_dur = int(sample_rate * 0.08)
        gap_dur = int(sample_rate * 0.05)
        total_dur = (pulse_dur + gap_dur) * 3
        t2 = np.linspace(0, total_dur / sample_rate, total_dur, endpoint=False)
        wave2 = np.zeros(total_dur, dtype=np.float32)
        for i in range(3):
            start = i * (pulse_dur + gap_dur)
            end = start + pulse_dur
            wave2[start:end] = np.sin(2 * np.pi * 1500 * t2[start:end])
        env2 = np.linspace(1.0, 0.3, total_dur)
        wave2 *= env2 * self.alert_volume
        sounds['risk_red'] = wave2

        # ML FALL: 下降音 (1000Hz → 300Hz)
        t3 = np.linspace(0, 0.5, int(sample_rate * 0.5), endpoint=False)
        freq = np.linspace(1000, 300, len(t3))
        phase = np.cumsum(2 * np.pi * freq / sample_rate)
        wave3 = np.sin(phase)
        env3 = np.linspace(1.0, 0.0, len(t3))
        wave3 *= env3 * self.alert_volume
        sounds['ml_fall'] = wave3

        # 转 pygame Sound 对象
        for key, wave in sounds.items():
            wave_int16 = (wave * 32767).astype(np.int16)
            stereo = np.column_stack([wave_int16, wave_int16])
            self._pygame_sounds[key] = pygame.sndarray.make_sound(stereo)

    def _play_alert(self, alert_level: str, ml_fall_triggered: bool):
        """播放告警音"""
        if _PYGAME_AVAILABLE and self._pygame_sounds:
            try:
                if ml_fall_triggered and 'ml_fall' in self._pygame_sounds:
                    self._pygame_sounds['ml_fall'].play()
                elif alert_level == "RED" and 'risk_red' in self._pygame_sounds:
                    self._pygame_sounds['risk_red'].play()
                elif 'risk_orange' in self._pygame_sounds:
                    self._pygame_sounds['risk_orange'].play()
                return
            except Exception:
                pass

        if _WINSOUND_AVAILABLE:
            # Windows 系统蜂鸣: 频率 1000Hz, 持续 300ms
            if alert_level == "RED":
                for _ in range(3):
                    winsound.Beep(1500, 150)
                    time.sleep(0.05)
            elif alert_level == "ORANGE":
                winsound.Beep(1000, 200)
                time.sleep(0.05)
                winsound.Beep(800, 200)
            else:
                winsound.Beep(1000, 300)

        # ════════════════════════════════════════════════════
        # 状态查询
        # ════════════════════════════════════════════════════

    def get_status(self) -> Dict:
        """获取当前告警系统状态"""
        now = time.time()
        cooldowns_remaining = {}
        for alert_type in ['risk_orange', 'risk_red', 'ml_fall']:
            last = self._cooldowns.get(alert_type, 0)
            cd = getattr(self, f'COOLDOWN_{alert_type.upper().replace("_", "_")}',
                        self.COOLDOWN_ORANGE if alert_type == 'risk_orange' else
                        self.COOLDOWN_RED if alert_type == 'risk_red' else
                        self.COOLDOWN_ML_FALL)
            remaining = max(0, cd - (now - last))
            if remaining > 0:
                cooldowns_remaining[alert_type] = remaining

        return {
            "enabled_audio": self.enable_audio,
            "enabled_screenshots": self.enable_screenshots,
            "total_alerts_fired": len(self.history),
            "cooldowns_remaining": cooldowns_remaining,
            "last_event": self.history[-1].iso_time if self.history else None,
            "audio_backend": "pygame" if (_PYGAME_AVAILABLE and self._pygame_sounds)
                             else "winsound" if _WINSOUND_AVAILABLE else "none",
        }

    def get_alert_counts(self, hours: int = 24) -> Dict[str, int]:
        """统计最近 N 小时的告警次数"""
        now = time.time()
        cutoff = now - hours * 3600
        counts = {"risk_orange": 0, "risk_red": 0, "ml_fall": 0}
        for evt in self.history:
            if evt.timestamp >= cutoff:
                for t in ['risk_orange', 'risk_red', 'ml_fall']:
                    if t in evt.alert_type:
                        counts[t] += 1
        return counts


# ════════════════════════════════════════════════════════
# 独立测试
# ════════════════════════════════════════════════════════

def main():
    """测试告警系统"""
    import argparse
    ap = argparse.ArgumentParser(description="Alert Manager — 告警系统测试")
    ap.add_argument("--dir", default=r"E:\老人跌倒\alerts", help="告警目录")
    ap.add_argument("--no-audio", action="store_true", help="禁用音频")
    ap.add_argument("--no-screenshot", action="store_true", help="禁用截图")
    args = ap.parse_args()

    mgr = AlertManager(
        alerts_dir=args.dir,
        enable_audio=not args.no_audio,
        enable_screenshots=not args.no_screenshot,
    )

    print(f"\n  告警系统状态: {mgr.get_status()}")
    print(f"\n  模拟告警测试 (连续 5 次)...")
    print(f"  {'─'*40}")

    # 模拟触发
    for i in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Test Frame {i}", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        evt = mgr.update(
            risk_score=65.0,
            alert_level="ORANGE" if i < 3 else "RED",
            ml_fall_triggered=(i == 4),
            fall_direction="backward",
            frame=frame,
        )

        if evt:
            print(f"  [{i}] 告警触发: type={evt.alert_type}, level={evt.alert_level}")
            if evt.screenshot_path:
                print(f"       截图: {os.path.basename(evt.screenshot_path)}")
        else:
            print(f"  [{i}] 冷却中, 未触发")

        time.sleep(0.3)

    print(f"\n  最终状态: {mgr.get_status()}")
    print(f"  最近 24h 告警: {mgr.get_alert_counts(24)}")
    print(f"\n  测试完成")


if __name__ == "__main__":
    import numpy as np
    main()
