#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_collector.py — 跌倒数据采集工具
=====================================
配合萤石 C6c / 本地摄像头录制标注视频片段。

操作:
    SPACE    — 开始/停止录制
    1-9      — 选择当前动作标签
    R        — 查看已录制的动作表
    Q / ESC  — 退出

录制后自动保存:
    videos/<标签>_<时间戳>.mp4        # 视频文件
    videos/labels.json                # 标注记录

用法:
    conda activate fall
    python data_collector.py                           # 使用本地摄像头
    python data_collector.py --rtsp "rtsp://..."        # 萤石 RTSP
    python data_collector.py --preview                  # 预览模式（不录制）
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from collections import OrderedDict
import cv2
import numpy as np

# ============================================================
# 动作标签定义
# ============================================================

ACTION_LABELS = OrderedDict([
    ("1", {"name": "forward_fall",    "cn": "正向跌倒",    "is_fall": True,  "color": (0, 0, 255)}),   # 红
    ("2", {"name": "sideways_fall",   "cn": "侧向跌倒",    "is_fall": True,  "color": (0, 0, 255)}),
    ("3", {"name": "backward_fall",   "cn": "后仰跌倒",    "is_fall": True,  "color": (0, 0, 255)}),
    ("4", {"name": "slip_fall",       "cn": "滑倒",       "is_fall": True,  "color": (0, 0, 255)}),
    ("5", {"name": "walking",         "cn": "行走",       "is_fall": False, "color": (0, 255, 0)}),   # 绿
    ("6", {"name": "bending",         "cn": "弯腰捡物",    "is_fall": False, "color": (0, 255, 0)}),
    ("7", {"name": "sitting_down",    "cn": "坐下",       "is_fall": False, "color": (0, 255, 0)}),
    ("8", {"name": "squatting",       "cn": "蹲下起立",    "is_fall": False, "color": (0, 255, 0)}),
    ("9", {"name": "lying_down",      "cn": "躺下",       "is_fall": False, "color": (0, 255, 255)}), # 黄
])

# ============================================================
# 采集器
# ============================================================

class DataCollector:
    def __init__(self, rtsp_url: str = None, output_dir: str = "E:/老人跌倒/data/collected/",
                 fps: float = 20.0, max_clip_seconds: int = 30):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.labels_file = os.path.join(output_dir, "labels.json")
        self.recording = False
        self.current_label = "1"
        self.writer = None
        self.record_start_time = 0.0
        self.max_clip_seconds = max_clip_seconds
        self.fps = fps
        self.clip_records = self._load_records()

        # 摄像头
        self._init_camera(rtsp_url)

    def _init_camera(self, rtsp_url: str = None):
        if rtsp_url:
            print(f"[CAM] 连接 RTSP: {rtsp_url[:60]}...")
            self.cap = cv2.VideoCapture(rtsp_url)
        else:
            print("[CAM] 打开本地摄像头...")
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头！请检查连接。")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  [OK] {self.width}x{self.height}")

    def _load_records(self) -> list:
        if os.path.exists(self.labels_file):
            with open(self.labels_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_records(self):
        with open(self.labels_file, "w", encoding="utf-8") as f:
            json.dump(self.clip_records, f, ensure_ascii=False, indent=2)

    def _start_recording(self):
        self.recording = True
        self.record_start_time = time.time()
        label_info = ACTION_LABELS[self.current_label]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label_info['name']}_{ts}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(filepath, fourcc, self.fps,
                                       (self.width, self.height))
        self.current_filename = filename
        self.current_filepath = filepath
        print(f"\n  [REC] 开始录制: {label_info['cn']} → {filename}")

    def _stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        duration = time.time() - self.record_start_time
        if self.writer:
            self.writer.release()
            self.writer = None

        label_info = ACTION_LABELS[self.current_label]
        record = {
            "filename": self.current_filename,
            "filepath": self.current_filepath,
            "label": label_info["name"],
            "label_cn": label_info["cn"],
            "is_fall": label_info["is_fall"],
            "duration_s": round(duration, 1),
            "timestamp": datetime.now().isoformat(),
        }
        self.clip_records.append(record)
        self._save_records()
        print(f"  [STOP] 完成: {round(duration, 1)}s | 总已录: {len(self.clip_records)} 段")

    def _draw_hud(self, frame, key_hint: str = ""):
        """绘制录制 UI"""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 顶部状态栏
        label_info = ACTION_LABELS[self.current_label]
        status_color = (0, 0, 255) if self.recording else (0, 255, 0)
        status_text = "● RECORDING" if self.recording else "○ STANDBY"

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        cv2.putText(frame, status_text, (10, 25), font, 0.6, status_color, 2)
        cv2.putText(frame, f"Action: [{self.current_label}] {label_info['cn']} ({label_info['name']})",
                    (10, 50), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Clips: {len(self.clip_records)} | "
                    f"Fall={sum(1 for r in self.clip_records if r['is_fall'])} "
                    f"ADL={sum(1 for r in self.clip_records if not r['is_fall'])}",
                    (10, 75), font, 0.45, (180, 180, 180), 1)

        # 底部帮助栏
        help_text = "SPACE=Rec | 1-9=Label | R=Show Table | Q=Quit"
        cv2.rectangle(frame, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, help_text, (10, h - 8), font, 0.45, (200, 200, 200), 1)

        if key_hint:
            cv2.putText(frame, key_hint, (w // 2 - 150, h // 2),
                        font, 0.8, (0, 255, 255), 2)

        # 录制计时
        if self.recording:
            elapsed = time.time() - self.record_start_time
            timer_text = f"{elapsed:.1f}s / {self.max_clip_seconds}s"
            cv2.circle(frame, (w - 40, 20), 8, (0, 0, 255), -1)
            cv2.putText(frame, timer_text, (w - 170, 25), font, 0.5, (0, 0, 255), 1)

        return frame

    def _print_table(self):
        """打印已录制动作统计表"""
        print(f"\n{'=' * 70}")
        print(f"  {'动作':<12} {'计划':>4}  {'已录':>4}  {'进度':>8}")
        print(f"  {'─' * 32}")

        plan = {
            "forward_fall": 5, "sideways_fall": 3, "backward_fall": 3,
            "slip_fall": 2, "walking": 5, "bending": 5,
            "sitting_down": 5, "squatting": 5, "lying_down": 3,
        }

        total_plan = 0
        total_done = 0
        for key, info in ACTION_LABELS.items():
            name = info["name"]
            target = plan.get(name, 3)
            done = sum(1 for r in self.clip_records if r["label"] == name)
            bar = "█" * done + "░" * (target - done) if target > 0 else ""
            print(f"  [{key}] {info['cn']:<10} {target:>4}  {done:>4}  {bar}")
            total_plan += target
            total_done += min(done, target)

        print(f"  {'─' * 32}")
        print(f"  {'合计':<12} {total_plan:>4}  {total_done:>4}")
        print(f"  {'=' * 70}")
        print(f"\n  跌倒类: {sum(1 for r in self.clip_records if r['is_fall'])} 段")
        print(f"  日常类: {sum(1 for r in self.clip_records if not r['is_fall'])} 段")
        print()

    def run(self):
        """主循环"""
        print("\n" + "=" * 60)
        print("  跌倒数据采集工具")
        print("=" * 60)
        self._print_table()
        print("  操作: SPACE=开始/停止 | 1-9=选标签 | R=看进度 | Q=退出\n")

        key_hint = ""
        key_hint_until = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # 录制
            if self.recording and self.writer:
                self.writer.write(frame)
                # 自动停止
                if time.time() - self.record_start_time > self.max_clip_seconds:
                    self._stop_recording()
                    key_hint = f"自动停止 ({self.max_clip_seconds}s 上限)"
                    key_hint_until = time.time() + 2

            # HUD
            frame = self._draw_hud(frame,
                                   key_hint if time.time() < key_hint_until else "")

            cv2.imshow("Data Collector | SPACE=Rec 1-9=Label Q=Quit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q or ESC
                if self.recording:
                    self._stop_recording()
                break
            elif key == ord(' '):  # SPACE
                if self.recording:
                    self._stop_recording()
                else:
                    self._start_recording()
            elif key == ord('r'):
                self._print_table()
            elif chr(key) in ACTION_LABELS:
                if not self.recording:
                    old_label = self.current_label
                    self.current_label = chr(key)
                    info = ACTION_LABELS[self.current_label]
                    key_hint = f"切换到: [{self.current_label}] {info['cn']}"
                    key_hint_until = time.time() + 1.5

        # 收尾
        self.cap.release()
        cv2.destroyAllWindows()

        print(f"\n{'=' * 60}")
        print(f"  采集结束")
        print(f"  总录制: {len(self.clip_records)} 段")
        print(f"  输出目录: {self.output_dir}")
        self._print_table()


def main():
    parser = argparse.ArgumentParser(description="跌倒数据采集工具")
    parser.add_argument("--rtsp", type=str, default=None,
                        help="RTSP 摄像头地址")
    parser.add_argument("--output", "-o", type=str,
                        default="E:/老人跌倒/data/collected/",
                        help="输出目录")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="录制帧率")
    parser.add_argument("--max_seconds", type=int, default=30,
                        help="单段最长秒数")
    args = parser.parse_args()

    collector = DataCollector(
        rtsp_url=args.rtsp,
        output_dir=args.output,
        fps=args.fps,
        max_clip_seconds=args.max_seconds,
    )
    collector.run()


if __name__ == "__main__":
    main()
