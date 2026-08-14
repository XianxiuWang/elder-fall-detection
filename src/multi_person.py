#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_person.py — 多人检测与跟踪模块 v1.0
==========================================
用途:
  1. YOLOv8n 行人检测 → 逐人 MediaPipe Pose 关键点
  2. IoU 行人跟踪 → 每人分配唯一 ID
  3. 每人的独立检测器状态管理

架构:
  摄像头/视频帧 → YOLOv8n 行人检测 → 逐人 bbox
       → IoU 匹配 (上一帧 → 当前帧) → 分配 person_id
       → 每人区域 crop → MediaPipe Pose → landmarks (33,4)
       → 返回 List[PersonResult]

用法:
    tracker = PersonTracker()
    persons = tracker.update(frame)  # → List[PersonResult]

    for p in persons:
        print(f"Person {p.person_id}: bbox={p.bbox}, 关键点={p.landmarks.shape}")
"""

import numpy as np
import cv2
import time
from typing import Optional, Dict, List, Tuple, Deque
from collections import deque
from dataclasses import dataclass, field

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[multi_person] YOLOv8 未安装 (ultralytics), 请先: pip install ultralytics")

import mediapipe as mp


# ════════════════════════════════════════════════════
# 数据类
# ════════════════════════════════════════════════════

@dataclass
class PersonResult:
    """单帧单人检测结果"""
    person_id: int                      # 行人唯一 ID
    bbox: Tuple[int, int, int, int]     # (x1, y1, x2, y2) 像素坐标
    bbox_conf: float = 1.0             # YOLO 检测置信度
    landmarks: Optional[np.ndarray] = None  # (33, 4) 关键点, [0,1] 归一化到全帧
    has_landmarks: bool = False
    mediapipe_ms: float = 0.0          # MediaPipe 耗时 (ms)

    # 按"关键点质量"排序用（可见关键点多→更可能是主要人物）
    @property
    def visible_keypoints(self) -> int:
        if self.landmarks is None:
            return 0
        return int(np.sum(self.landmarks[:, 3] > 0.5))


# ════════════════════════════════════════════════════
# 行人跟踪器
# ════════════════════════════════════════════════════

class PersonTrackerLegacy:
    """YOLOv8n + IoU 行人跟踪 + 逐人 MediaPipe Pose"""

    # 跟踪参数
    MAX_DISAPPEARED = 15       # 丢失多少帧后移除
    MIN_IOU_MATCH = 0.40       # IoU 匹配阈值
    MAX_PERSONS = 6            # 最多同时跟踪人数

    # MediaPipe 参数
    MP_COMPLEXITY = 1          # 模型复杂度 (0/1/2)
    MP_SMOOTH = True

    def __init__(self, yolo_model: str = None,
                 confidence: float = 0.4,
                 image_size: int = 640):
        """
        Args:
            yolo_model: YOLO 模型路径 (默认自动查找缓存/本地文件)
            confidence: YOLO 检测置信度阈值
            image_size: YOLO 输入尺寸
        """
        if not _YOLO_AVAILABLE:
            raise ImportError("ultralytics 未安装: pip install ultralytics")

        # ── 自动查找模型文件 ──
        if yolo_model is None:
            import pathlib
            candidates = [
                pathlib.Path.home() / '.cache' / 'torch' / 'hub' / 'ultralytics_yolov8_master' / 'yolov8n.pt',
                pathlib.Path(__file__).resolve().parent.parent / 'yolov8n.pt',
                pathlib.Path('yolov8n.pt'),
            ]
            yolo_model = "yolov8n.pt"  # fallback
            for p in candidates:
                if p.exists():
                    yolo_model = str(p)
                    break

        # ── YOLO 行人检测 ──
        print(f"[PersonTracker] 加载 YOLO 模型: {yolo_model}")
        self.yolo = YOLO(yolo_model)
        self.yolo_conf = confidence
        self.yolo_imgsz = image_size
        self.yolo_infer_ms = 0.0

        # ── MediaPipe Pose (每个行人独立推理) ──
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.MP_COMPLEXITY,
            smooth_landmarks=self.MP_SMOOTH,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ── 跟踪状态 ──
        self._next_id = 0
        self._tracked: Dict[int, Dict] = {}  # {person_id: {bbox, disappeared, ...}}
        self._total_detected = 0

    def update(self, frame: np.ndarray) -> List[PersonResult]:
        """
        处理一帧，返回所有检测到的行人结果。

        Args:
            frame: BGR 图像 (H, W, 3)

        Returns:
            List[PersonResult] 按 visible_keypoints 降序排列
        """
        h, w = frame.shape[:2]
        results = []

        # ═══ 1. YOLO 行人检测 ═══
        t0 = time.time()
        yolo_results = self.yolo(frame, classes=[0], conf=self.yolo_conf,
                                  imgsz=self.yolo_imgsz, verbose=False)
        self.yolo_infer_ms = (time.time() - t0) * 1000

        boxes_data = []
        if yolo_results and len(yolo_results) > 0:
            for box in yolo_results[0].boxes:
                if int(box.cls[0]) != 0:  # 只取 person 类
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                boxes_data.append((int(x1), int(y1), int(x2), int(y2), conf))

        # 截断到最大人数
        if len(boxes_data) > self.MAX_PERSONS:
            boxes_data.sort(key=lambda b: b[4], reverse=True)
            boxes_data = boxes_data[:self.MAX_PERSONS]

        # ═══ 2. IoU 匹配 (当前检测 ↔ 已跟踪行人) ═══
        matched_ids = self._match_detections(boxes_data)

        # 增加消失计数
        for pid in list(self._tracked.keys()):
            if pid not in matched_ids.values():
                self._tracked[pid]["disappeared"] += 1

        # 清理长时间消失的
        to_remove = [pid for pid, t in self._tracked.items()
                      if t["disappeared"] > self.MAX_DISAPPEARED]
        for pid in to_remove:
            del self._tracked[pid]

        # ═══ 3. 逐人 MediaPipe Pose ═══
        for (x1, y1, x2, y2, conf), person_id in matched_ids.items():
            # 更新跟踪信息
            self._tracked[person_id]["bbox"] = (x1, y1, x2, y2)
            self._tracked[person_id]["disappeared"] = 0
            self._tracked[person_id]["conf"] = conf

            # 提取关键点 (如果区域足够大)
            landmarks = None
            has_lm = False
            mp_ms = 0.0

            box_w = x2 - x1
            box_h = y2 - y1
            if box_w > 20 and box_h > 40:
                # 扩展 bbox 10% 给 MediaPipe 留出头部和脚部空间
                pad_x = int(box_w * 0.1)
                pad_y = int(box_h * 0.1)
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w, x2 + pad_x)
                cy2 = min(h, y2 + pad_y)

                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    t_mp = time.time()
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    mp_results = self.pose.process(crop_rgb)
                    mp_ms = (time.time() - t_mp) * 1000

                    if mp_results.pose_landmarks:
                        landmarks = np.zeros((33, 4), dtype=np.float32)
                        ch, cw = crop.shape[:2]
                        for i, lm_px in enumerate(mp_results.pose_landmarks.landmark):
                            # 从 crop 坐标映射回全帧归一化坐标 [0,1]
                            landmarks[i] = [
                                (lm_px.x * cw + cx1) / w,
                                (lm_px.y * ch + cy1) / h,
                                lm_px.z,
                                lm_px.visibility,
                            ]
                        has_lm = True

            if landmarks is None:
                landmarks = np.zeros((33, 4), dtype=np.float32)

            results.append(PersonResult(
                person_id=person_id,
                bbox=(x1, y1, x2, y2),
                bbox_conf=conf,
                landmarks=landmarks,
                has_landmarks=has_lm,
                mediapipe_ms=mp_ms,
            ))

        # ═══ 4. 排序: 关键点可见度高的排前面 ═══
        results.sort(key=lambda r: r.visible_keypoints, reverse=True)
        return results

    def _match_detections(self, boxes_data: list) -> Dict[Tuple, int]:
        """
        IoU 匹配：将当前检测框分配给已有的 tracked persons。

        Returns:
            {(x1,y1,x2,y2,conf): person_id, ...}
        """
        if not boxes_data:
            return {}

        # 上一帧的跟踪框
        prev_boxes = {pid: t["bbox"] for pid, t in self._tracked.items()
                       if t["disappeared"] <= 1}

        if not prev_boxes:
            # 全新：所有检测都是新行人
            result = {}
            for box in boxes_data:
                person_id = self._next_id
                self._next_id += 1
                self._tracked[person_id] = {"bbox": box[:4], "disappeared": 0, "conf": box[4]}
                result[box] = person_id
            return result

        # IoU 矩阵
        n_det = len(boxes_data)
        prev_ids = list(prev_boxes.keys())
        iou_matrix = np.zeros((n_det, len(prev_ids)), dtype=np.float32)

        for i, (dx1, dy1, dx2, dy2, _) in enumerate(boxes_data):
            for j, pid in enumerate(prev_ids):
                px1, py1, px2, py2 = prev_boxes[pid]
                iou_matrix[i, j] = self._iou(dx1, dy1, dx2, dy2, px1, py1, px2, py2)

        # 贪心匹配 (每行取最大 IoU, 依次分配)
        matched_det = set()
        matched_prev = set()
        result = {}

        # 先匹配高 IoU (≥ MIN_IOU_MATCH)
        for _ in range(min(n_det, len(prev_ids))):
            best_iou = self.MIN_IOU_MATCH
            best_i, best_j = -1, -1
            for i in range(n_det):
                if i in matched_det:
                    continue
                for j in range(len(prev_ids)):
                    if j in matched_prev:
                        continue
                    if iou_matrix[i, j] > best_iou:
                        best_iou = iou_matrix[i, j]
                        best_i, best_j = i, j

            if best_i >= 0:
                matched_det.add(best_i)
                matched_prev.add(best_j)
                result[boxes_data[best_i]] = prev_ids[best_j]

        # 剩余未匹配的检测 → 新行人
        for i in range(n_det):
            if i not in matched_det:
                person_id = self._next_id
                self._next_id += 1
                box = boxes_data[i]
                self._tracked[person_id] = {"bbox": box[:4], "disappeared": 0, "conf": box[4]}
                result[box] = person_id

        return result

    @staticmethod
    def _iou(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b) -> float:
        """计算两个边界框的 IoU"""
        xi1 = max(x1a, x1b)
        yi1 = max(y1a, y1b)
        xi2 = min(x2a, x2b)
        yi2 = min(y2a, y2b)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area_a = (x2a - x1a) * (y2a - y1a)
        area_b = (x2b - x1b) * (y2b - y1b)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def get_status(self) -> dict:
        """获取当前跟踪状态"""
        active = sum(1 for t in self._tracked.values() if t["disappeared"] <= 1)
        return {
            "total_detected": self._total_detected,
            "tracked_count": len(self._tracked),
            "active_count": active,
            "yolo_ms": round(self.yolo_infer_ms, 1),
        }

    def close(self):
        """释放资源"""
        try:
            self.pose.close()
        except (ValueError, AttributeError):
            pass


# ════════════════════════════════════════════════════
# 稳定跟踪 (v2.0)
# ════════════════════════════════════════════════════
# 默认使用 stable_tracker.StablePersonTracker 作为 PersonTracker 实现，
# 解决跟踪 ID 碎片化。原 v1.0 实现保留为 PersonTrackerLegacy 可回退。
try:
    from .stable_tracker import StablePersonTracker, PersonResult
    # 别名: e2e 引用 multi_person.PersonTracker 时自动使用稳定版
    PersonTracker = StablePersonTracker
except Exception as _e:  # noqa: F841
    print(f"[multi_person] stable_tracker 导入失败, 回退到 PersonTrackerLegacy: {_e}")
    PersonTracker = PersonTrackerLegacy


# ════════════════════════════════════════════════════
# 测试入口
# ════════════════════════════════════════════════════

def main():
    """多人检测跟踪独立测试"""
    import argparse
    ap = argparse.ArgumentParser(description="Person Tracker 测试")
    ap.add_argument("--source", "-s", type=str, default=None,
                     help="视频文件路径 (默认摄像头)")
    ap.add_argument("--conf", type=float, default=0.4,
                     help="YOLO 检测置信度阈值")
    args = ap.parse_args()

    tracker = PersonTracker(confidence=args.conf)

    if args.source:
        cap = cv2.VideoCapture(args.source)
    else:
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开视频源")
        return

    print(f"\n  按 Q 退出 | 跟踪中...\n")
    frame_idx = 0
    fps_window = deque(maxlen=30)
    person_colors = {}  # {person_id: BGR color}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.time()
        persons = tracker.update(frame)
        elapsed = (time.time() - t_start) * 1000
        fps_window.append(1.0 / max(time.time() - t_start, 0.001))
        fps = sum(fps_window) / len(fps_window) if fps_window else 0

        # 绘制
        for p in persons:
            # 分配颜色
            if p.person_id not in person_colors:
                import random
                person_colors[p.person_id] = (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255),
                )
            color = person_colors[p.person_id]

            # 边界框
            x1, y1, x2, y2 = p.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{p.person_id} [{p.visible_keypoints}kp]"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 关键点
            if p.has_landmarks and p.landmarks is not None:
                h, w = frame.shape[:2]
                for i in range(33):
                    if p.landmarks[i, 3] > 0.5:  # 可见
                        px = int(p.landmarks[i, 0] * w)
                        py = int(p.landmarks[i, 1] * h)
                        cv2.circle(frame, (px, py), 2, color, -1)

        # 帧信息
        status = tracker.get_status()
        cv2.putText(frame,
                    f"F:{frame_idx} | YOLO:{status['yolo_ms']}ms | "
                    f"FPS:{fps:.0f} | Persons:{status['active_count']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Person Tracker Test", frame)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
