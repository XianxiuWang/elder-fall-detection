"""
护龄 — 实时推理演示
====================
完整两阶段流水线的实时推理：
  阶段1: YOLO11n → 检测画面中的人体，提取 bbox + 姿态关键点
  阶段2: 微调后的分类模型 → 判断人的状态（行走/坐/躺/跌倒…）

画面叠加显示：
  · 人体边界框 + 状态标签
  · 17点姿态骨架
  · 实时 FPS
  · 状态历史时间线

用法：
    # 使用摄像头实时推理
    python inference_demo.py

    # 使用视频文件
    python inference_demo.py --video D:/videos/test.mp4

    # 指定模型
    python inference_demo.py --detector yolo11n.pt --classifier models/best_model.pth

    # 保存推理结果视频
    python inference_demo.py --output demo_output.mp4
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

import config


# ============================================================
# 颜色方案
# ============================================================
COLORS = {
    'green':  (0, 255, 0),
    'red':    (0, 0, 255),
    'yellow': (0, 255, 255),
    'orange': (0, 165, 255),
    'blue':   (255, 0, 0),
    'white':  (255, 255, 255),
    'black':  (0, 0, 0),
    'cyan':   (255, 255, 0),
}

ALERT_COLORS = {
    0: COLORS['green'],    # 行走
    1: COLORS['green'],    # 坐着
    2: COLORS['green'],    # 躺卧
    3: COLORS['yellow'],   # 久坐
    4: COLORS['orange'],   # 异常
    5: COLORS['red'],      # 跌倒
    6: COLORS['red'],      # 无人
}


# ============================================================
# 推理流水线
# ============================================================

class PersonStateClassifier:
    """
    两阶段人体状态分类器

    Stage 1: YOLO11n 检测人体
    Stage 2: 微调分类模型判断状态
    """

    def __init__(self, detector_model: str, classifier_model: str,
                 device: str = None):
        if device is None:
            device = config.DEVICE
        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available()
            else "mps" if device == "mps" and torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"💻 推理设备: {self.device}")

        # ── 阶段1: 加载 YOLO 检测器 ──
        print(f"🔍 加载检测器: {detector_model}")
        from ultralytics import YOLO
        self.detector = YOLO(detector_model)
        self.detector.to(self.device)

        # ── 阶段2: 加载分类器 ──
        print(f"🧠 加载分类器: {classifier_model}")
        self.classifier = self._load_classifier(classifier_model)

        # 预处理（与训练时保持一致）
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.CLASSIFIER_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # 时序平滑（最近N帧的投票）
        self.temporal_window = config.TEMPORAL_WINDOW
        self.prediction_history = deque(maxlen=self.temporal_window)

        print(f"✅ 推理引擎就绪\n")

    def _load_classifier(self, checkpoint_path: str) -> nn.Module:
        """加载微调后的分类模型"""
        from train_classifier import build_model

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 从 checkpoint 获取模型信息
        model_name = ckpt.get('model_name', config.DEFAULT_CLASSIFIER)
        num_classes = ckpt.get('num_classes', config.NUM_CLASSES)

        model = build_model(model_name, num_classes)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        print(f"   模型: {model_name}, {num_classes}类")
        if 'val_acc' in ckpt:
            print(f"   验证精度: {ckpt['val_acc']*100:.1f}% | F1: {ckpt.get('val_f1', 0)*100:.1f}%")

        return model

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> dict:
        """
        单帧推理

        Args:
            frame: BGR 图像 (H, W, 3)

        Returns:
            {
                'has_person': bool,
                'bbox': (x1, y1, x2, y2) or None,
                'state_id': int,
                'state_name': str,
                'confidence': float,
                'keypoints': np.array or None,
                'alert_level': str,
            }
        """
        h, w = frame.shape[:2]

        # ──── 阶段1: YOLO 检测人体 ────
        det_results = self.detector(
            frame,
            conf=config.DETECTION_CONF,
            iou=config.DETECTION_IOU,
            classes=[config.PERSON_CLASS_ID],  # 只检测 person
            verbose=False,
        )

        if len(det_results) == 0 or det_results[0].boxes is None or \
           len(det_results[0].boxes) == 0:
            # 无人
            self.prediction_history.append(6)
            return self._format_result(None, 6)

        # 取置信度最高的人体
        boxes = det_results[0].boxes
        confs = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        best_conf = float(confs[best_idx])

        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # ──── 阶段2: 分类状态 ────
        if x2 <= x1 or y2 <= y1:
            self.prediction_history.append(6)
            return self._format_result(None, 6)

        # 裁剪人体区域
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            self.prediction_history.append(6)
            return self._format_result(None, 6)

        # 预处理 + 推理
        input_tensor = self.transform(crop).unsqueeze(0).to(self.device)
        output = self.classifier(input_tensor)
        probs = torch.softmax(output, dim=1)
        state_id = int(probs.argmax(dim=1).item())
        state_conf = float(probs.max().item())

        # 时序平滑
        self.prediction_history.append(state_id)
        smoothed_state = self._temporal_smooth()

        return self._format_result(
            (x1, y1, x2, y2), smoothed_state, best_conf, state_conf
        )

    def _temporal_smooth(self) -> int:
        """时序多数投票平滑"""
        if len(self.prediction_history) == 0:
            return 1  # 默认坐着

        # 多数投票
        votes = {}
        for s in self.prediction_history:
            votes[s] = votes.get(s, 0) + 1

        # 取票数最多的状态
        best_state = max(votes, key=votes.get)

        # 特殊规则：如果最近一帧是跌倒，且票数≥2，则判定为跌倒（降低漏报）
        recent = list(self.prediction_history)[-3:]
        if 5 in recent and recent.count(5) >= 2:
            return 5

        return best_state

    def _format_result(self, bbox, state_id, det_conf=None, cls_conf=None):
        return {
            'has_person': bbox is not None,
            'bbox': bbox,
            'state_id': state_id,
            'state_name': config.STATE_NAMES.get(state_id, '未知'),
            'state_name_en': config.STATE_NAMES_EN.get(state_id, 'unknown'),
            'detection_confidence': det_conf,
            'classification_confidence': cls_conf,
            'alert_level': config.ALERT_LEVEL.get(state_id, ('', ''))[0],
            'alert_color': ALERT_COLORS.get(state_id, COLORS['white']),
        }


# ============================================================
# 可视化
# ============================================================

class Visualizer:
    """画面叠加绘制"""

    def __init__(self):
        self.fps_history = deque(maxlen=30)
        self.state_history = deque(maxlen=100)  # 状态时间线
        self.last_time = time.time()

    def draw(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """在画面和帧上叠加所有信息"""
        h, w = frame.shape[:2]
        current_time = time.time()
        fps = 1.0 / max(current_time - self.last_time, 0.001)
        self.fps_history.append(fps)
        self.last_time = current_time
        avg_fps = np.mean(self.fps_history)

        # ── 绘制人体框 + 状态标签 ──
        if result['has_person'] and result['bbox']:
            x1, y1, x2, y2 = result['bbox']
            color = result['alert_color']

            # 边框
            thickness = 3 if result['state_id'] == 5 else 2  # 跌倒时加粗
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 标签背景
            label_text = f"{result['state_name']} ({result['classification_confidence']*100:.0f}%)"
            (label_w, label_h), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - 10),
                (x1 + label_w + 10, y1),
                color, -1
            )
            cv2.putText(
                frame, label_text,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['white'], 2
            )

        # 记录状态历史
        timestamp = datetime.now().strftime("%H:%M:%S")
        alert_level, _ = config.ALERT_LEVEL.get(result['state_id'], ('', ''))
        self.state_history.append((timestamp, result['state_name'], alert_level))

        # ── 右侧信息面板 ──
        panel_x = w - 220
        panel_w = 220

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, 0), (w, h), (20, 20, 20), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        y_offset = 30

        # FPS
        fps_text = f"FPS: {avg_fps:.1f}"
        cv2.putText(frame, fps_text, (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['cyan'], 2)
        y_offset += 30

        # 分隔线
        cv2.line(frame, (panel_x + 5, y_offset), (panel_x + panel_w - 5, y_offset),
                 COLORS['white'], 1)
        y_offset += 15

        # 状态信息
        cv2.putText(frame, "当前状态:", (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
        y_offset += 25

        status_color = result['alert_color']
        cv2.putText(frame, f"  {result['state_name']}",
                    (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        y_offset += 30

        if result['classification_confidence']:
            cv2.putText(frame, f"  置信度: {result['classification_confidence']*100:.0f}%",
                        (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
            y_offset += 20

        if result['detection_confidence']:
            cv2.putText(frame, f"  检测度: {result['detection_confidence']*100:.0f}%",
                        (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['white'], 1)
            y_offset += 30

        # 告警级别
        cv2.line(frame, (panel_x + 5, y_offset), (panel_x + panel_w - 5, y_offset),
                 COLORS['white'], 1)
        y_offset += 15

        cv2.putText(frame, f"告警: {result['alert_level']}",
                    (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # ── 底部状态时间线 ──
        timeline_y = h - 40
        cv2.rectangle(frame, (0, timeline_y), (panel_x, h), (30, 30, 30), -1)

        if len(self.state_history) > 1:
            # 显示最近5条
            recent = list(self.state_history)[-5:]
            text = " | ".join(
                f"{ts} {state}" for ts, state, _ in recent
            )
            cv2.putText(frame, text, (5, timeline_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['white'], 1)

        return frame


# ============================================================
# 主循环
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="护龄 实时推理演示")
    parser.add_argument("--video", type=str, default=None,
                        help="输入视频路径（不填则使用摄像头）")
    parser.add_argument("--detector", type=str, default=config.DETECTION_MODEL,
                        help="YOLO检测模型")
    parser.add_argument("--classifier", type=str,
                        default=str(config.MODEL_DIR / "best_model.pth"),
                        help="分类模型 checkpoint")
    parser.add_argument("--output", type=str, default=None,
                        help="输出视频路径（保存推理结果）")
    parser.add_argument("--device", type=str, default=config.DEVICE,
                        help="推理设备")
    parser.add_argument("--no_display", action="store_true",
                        help="不显示画面（配合--output使用）")
    args = parser.parse_args()

    # 检查分类模型是否存在
    classifier_path = Path(args.classifier)
    if not classifier_path.exists():
        print(f"❌ 分类模型不存在: {classifier_path}")
        print("请先运行: python train_classifier.py")
        return

    # 初始化
    pipeline = PersonStateClassifier(
        args.detector, str(classifier_path), args.device
    )
    viz = Visualizer()

    # 视频源
    if args.video:
        cap = cv2.VideoCapture(args.video)
        input_name = Path(args.video).name
    else:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        input_name = "摄像头"

    if not cap.isOpened():
        print(f"❌ 无法打开 {'视频: ' + args.video if args.video else '摄像头'}")
        return

    fps_input = cap.get(cv2.CAP_PROP_FPS)
    print(f"📹 输入: {input_name} ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, {fps_input:.0f}fps)")

    # 输出视频
    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter(args.output, fourcc, 15.0, (frame_w, frame_h))
        print(f"💾 输出: {args.output}")

    print(f"\n按 Q 退出 | 按 S 截图\n")
    print(f"{'='*60}")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 推理
            result = pipeline.predict(frame)

            # 可视化
            display_frame = viz.draw(frame.copy(), result)

            # 输出（可选）
            if out_writer:
                out_writer.write(display_frame)

            # 显示
            if not args.no_display:
                cv2.imshow("护龄 v3 - 实时状态分类", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot = f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(screenshot, display_frame)
                print(f"📸 截图已保存: {screenshot}")

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  已处理 {frame_count} 帧 | {frame_count/elapsed:.1f} FPS | "
                      f"当前: {result['state_name']}")

    except KeyboardInterrupt:
        print("\n⏹ 用户中断")
    finally:
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"📊 推理统计:")
        print(f"   总帧数: {frame_count}")
        print(f"   总耗时: {elapsed:.1f}s")
        print(f"   平均FPS: {frame_count/elapsed:.1f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
