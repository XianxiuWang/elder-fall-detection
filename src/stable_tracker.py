#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stable_tracker.py — 多人检测 + 稳定跟踪模块 v2.0
=================================================
在 multi_person.py (v1.0) 基础上，重点解决"跟踪 ID 碎片化"问题。

v2.0 相对 v1.0 的核心改进:
  1. 扩大匹配池: 不再只匹配"上一帧", 而是包含最近 GRACE_FRAMES 帧内出现过的轨,
     避免单帧漏检(YOLO 不稳定)造成的 ID 断裂。
  2. 位置 + 外观联合打分: IoU 之外, 引入中心距离(Gaussian 先验)、bbox 面积比、
     LAB 颜色直方图(外观特征), 综合匹配。对遮挡/摔倒时 bbox 剧烈变化更鲁棒。
  3. 软重识别 (soft re-ID): 为每轨保存人体区域外观特征; 新检测无法匹配活跃轨时,
     在"近期消失池 + 历史 reuse 池"中按外观相似度找回原 ID, 而不是立即分配新 ID。
  4. ID 复用池: 被清理的轨保留外观特征(带时间戳)进入 reuse pool, 新检测优先重识别。
  5. 卡尔曼预测(轻量匀速): 检测缺失时用预测框参与匹配, 桥接暂时遮挡造成的断裂。

接口与 v1.0 完全兼容 (update / get_status / close / PersonResult),
可直接替换 multi_person.py 使用。
"""

import numpy as np
import cv2
import time
from typing import Optional, Dict, List, Tuple
from collections import deque
from dataclasses import dataclass, field

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[stable_tracker] YOLOv8 未安装 (ultralytics), 请先: pip install ultralytics")

import mediapipe as mp


# ════════════════════════════════════════════════════
# 数据类
# ════════════════════════════════════════════════════

@dataclass
class PersonResult:
    """单帧单人检测结果 (与 multi_person 兼容)"""
    person_id: int
    bbox: Tuple[int, int, int, int]
    bbox_conf: float = 1.0
    landmarks: Optional[np.ndarray] = None
    has_landmarks: bool = False
    mediapipe_ms: float = 0.0

    @property
    def visible_keypoints(self) -> int:
        if self.landmarks is None:
            return 0
        return int(np.sum(self.landmarks[:, 3] > 0.5))


# ════════════════════════════════════════════════════
# 轻量匀速卡尔曼 (用于桥接暂时跟踪断裂)
# ════════════════════════════════════════════════════

class _BoxKF:
    """极简匀速模型: 4 个独立的 1-D 常数速度卡尔曼, 无外部依赖。
    每个坐标轴独立: 状态 [x, v], 常数速度预测, 位置观测更新。
    """

    def __init__(self, box: Tuple[int, int, int, int]):
        b = np.array(box, dtype=np.float64)
        # 每轴: [pos, vel, P]
        self.states = np.zeros((4, 3), dtype=np.float64)
        self.states[:, 0] = b            # pos
        self.states[:, 2] = 50.0         # P
        self.R = 8.0        # 测量噪声
        self.Q = 6.0        # 过程噪声

    def predict(self):
        """预测下一帧状态, 返回预测框 (pos + vel)"""
        pos = self.states[:, 0] + self.states[:, 1]
        P = self.states[:, 2] + self.Q
        self.states[:, 0] = pos
        self.states[:, 2] = P
        return tuple(int(round(v)) for v in pos)

    def update(self, box: Tuple[int, int, int, int]):
        """用新测量更新状态 (标准标量卡尔曼, 无 NaN 风险)"""
        z = np.array(box, dtype=np.float64)
        P = self.states[:, 2]
        K = P / (P + self.R)                       # 标量增益
        innov = z - self.states[:, 0]
        # 更新 pos
        self.states[:, 0] = self.states[:, 0] + K * innov
        # 更新 vel (用创新项)
        self.states[:, 1] = self.states[:, 1] + K * innov
        # 更新 P
        self.states[:, 2] = (1.0 - K) * P

    def box(self) -> Tuple[int, int, int, int]:
        return tuple(int(round(v)) for v in self.states[:, 0])


# ════════════════════════════════════════════════════
# 外观特征 / 匹配工具
# ════════════════════════════════════════════════════

HIST_BINS = 32


def _extract_appearance(frame: np.ndarray,
                        box: Tuple[int, int, int, int],
                        hist_size: int = HIST_BINS) -> Optional[np.ndarray]:
    """
    提取人体区域的外观特征: LAB 空间 3 通道扁平直方图 (归一化)。
    返回 shape (hist_size*3,) 的 float32 向量; 区域无效时返回 None。
    """
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1)); x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1)); y2 = max(y1 + 1, min(y2, h))
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return None
    roi = frame[y1:y2, x1:x2]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    feats = []
    # 只取中央 70% 区域, 减少背景干扰
    ch, cw = lab.shape[:2]
    yy0, yy1 = int(ch * 0.10), int(ch * 0.75)
    xx0, xx1 = int(cw * 0.15), int(cw * 0.85)
    if yy1 > yy0 and xx1 > xx0:
        lab = lab[yy0:yy1, xx0:xx1]
    for c in range(3):
        hist = cv2.calcHist([lab], [c], None, [hist_size], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        feats.append(hist)
    if not feats:
        return None
    feat = np.concatenate(feats).astype(np.float32)
    n = np.linalg.norm(feat)
    if n > 0:
        feat /= n
    return feat


def _appearance_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """外观余弦相似度 [0,1]; 任一缺失返回 0。"""
    if a is None or b is None:
        return 0.0
    return max(0.0, float(np.dot(a, b)))


# ════════════════════════════════════════════════════
# 稳定行人跟踪器
# ════════════════════════════════════════════════════

class StablePersonTracker:
    """YOLOv8n + 稳定跟踪 (IoU + 外观 + 卡尔曼) + 逐人 MediaPipe"""

    # ── 跟踪参数 ──
    MAX_DISAPPEARED = 30        # 活跃轨: 丢失多少帧后从活跃池移出(但进 reuse 池)
    GRACE_FRAMES = 12           # 匹配池内含最近多少帧出现过的"短暂消失"轨
    MIN_IOU_MATCH = 0.35        # IoU 匹配底线 (匹配池内)
    MAX_PERSONS = 6            # 最多同时跟踪人数

    # 联合打分权重 (位置 + 外观)
    W_IOU = 0.55               # IoU 权重
    W_CENTER = 0.20            # 中心距离权重
    W_SIZE = 0.10             # 尺寸比权重
    W_APP = 0.15               # 外观权重
    MATCH_JOINT_THRESH = 0.35  # 联合得分匹配阈值

    # 软重识别
    REUSE_POOL_LIFETIME = 300  # reuse 池内 ID 保留帧数(约10秒)
    REID_MIN_IOU = 0.02        # re-ID 时 IoU 的最低要求 (位置合理性)
    REID_MIN_APP = 0.72        # re-ID 时外观相似度阈值
    REID_MAX_CENTER = 0.40     # re-ID 时中心距离(归一化)上限

    # MediaPipe
    MP_COMPLEXITY = 1
    MP_SMOOTH = True

    def __init__(self, yolo_model: str = None,
                 confidence: float = 0.4,
                 image_size: int = 640):
        if not _YOLO_AVAILABLE:
            raise ImportError("ultralytics 未安装: pip install ultralytics")

        # ── 模型查找 (同 v1.0) ──
        if yolo_model is None:
            import pathlib
            candidates = [
                pathlib.Path.home() / '.cache' / 'torch' / 'hub' / 'ultralytics_yolov8_master' / 'yolov8n.pt',
                pathlib.Path(__file__).resolve().parent.parent / 'yolov8n.pt',
                pathlib.Path('yolov8n.pt'),
            ]
            yolo_model = "yolov8n.pt"
            for p in candidates:
                if p.exists():
                    yolo_model = str(p)
                    break

        print(f"[StablePersonTracker] 加载 YOLO 模型: {yolo_model}")
        self.yolo = YOLO(yolo_model)
        self.yolo_conf = confidence
        self.yolo_imgsz = image_size
        self.yolo_infer_ms = 0.0

        # MediaPipe
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
        self._frame_count = 0
        # 活跃轨: {pid: {bbox, disappeared, conf, kf, appearance, last_bbox}}
        self._tracked: Dict[int, Dict] = {}
        # 近期消失轨(在匹配池内): 由 _tracked 中 disappeared<=GRACE_FRAMES 选出
        # reuse 池(身份记忆): {pid: {appearance, last_seen_frame, last_bbox}}
        self._reuse_pool: Dict[int, Dict] = {}
        self._total_detected = 0
        # 统计用
        self.id_frame_counts: Dict[int, int] = {}   # 每个 ID 累计出现的帧数

    # ──────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────
    # (KF 用 4 个独立的 1-D 常数速度滤波器, 仅在 _BoxKF 内部使用, 无外部依赖)

    def update(self, frame: np.ndarray) -> List[PersonResult]:
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
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                boxes_data.append((int(x1), int(y1), int(x2), int(y2), conf))

        if len(boxes_data) > self.MAX_PERSONS:
            boxes_data.sort(key=lambda b: b[4], reverse=True)
            boxes_data = boxes_data[:self.MAX_PERSONS]

        # ═══ 2. 稳定匹配 (联合打分) ═══
        matched_boxes = self._match_detections(boxes_data, frame, h, w)

        # matched_boxes: {(x1,y1,x2,y2,conf): person_id}
        # 更新消失计数
        for pid in list(self._tracked.keys()):
            if pid not in matched_boxes.values():
                self._tracked[pid]["disappeared"] += 1
                # 卡尔曼预测, 保留预测框
                if self._tracked[pid]["kf"] is not None:
                    self._tracked[pid]["pred_box"] = self._tracked[pid]["kf"].predict()

        # 清理: 超过 MAX_DISAPPEARED 的活跃轨 → 进入 reuse 池
        for pid in list(self._tracked.keys()):
            if self._tracked[pid]["disappeared"] > self.MAX_DISAPPEARED:
                t = self._tracked.pop(pid)
                # 身份记忆进 reuse 池
                if t.get("appearance") is not None:
                    self._reuse_pool[pid] = {
                        "appearance": t["appearance"],
                        "last_seen_frame": self._frame_count,
                        "last_bbox": t.get("last_bbox", t["bbox"]),
                    }

        # ═══ 3. 逐人 MediaPipe + 结果组装 ═══
        for (x1, y1, x2, y2, conf), pid in matched_boxes.items():
            box = (x1, y1, x2, y2)
            self._tracked[pid]["bbox"] = box
            self._tracked[pid]["last_bbox"] = box
            self._tracked[pid]["disappeared"] = 0
            self._tracked[pid]["conf"] = conf
            # 更新卡尔曼 + 外观特征 (在本帧 ROI 上)
            if self._tracked[pid]["kf"] is not None:
                self._tracked[pid]["kf"].update(box)
            app = _extract_appearance(frame, box)
            if app is not None:
                # 外观 EMA 平滑, 抵抗噪声
                old = self._tracked[pid].get("appearance")
                if old is not None:
                    merged = 0.6 * old + 0.4 * app
                    n = np.linalg.norm(merged)
                    if n > 0:
                        merged /= n
                    self._tracked[pid]["appearance"] = merged
                else:
                    self._tracked[pid]["appearance"] = app
            self._tracked[pid]["pred_box"] = None

            # 统计
            self.id_frame_counts[pid] = self.id_frame_counts.get(pid, 0) + 1

            # 关键点
            landmarks, has_lm, mp_ms = self._pose_on_box(frame, box, h, w)
            if landmarks is None:
                landmarks = np.zeros((33, 4), dtype=np.float32)
            results.append(PersonResult(
                person_id=pid, bbox=box, bbox_conf=conf,
                landmarks=landmarks, has_landmarks=has_lm,
                mediapipe_ms=mp_ms,
            ))

        # 更新总检测数 (去重: 统计本帧不重复的活跃人数)
        self._total_detected = max(self._total_detected, len(self._tracked))

        # ═══ 4. 清理过期的 reuse 池 ═══
        cutoff = self._frame_count - self.REUSE_POOL_LIFETIME
        for pid in list(self._reuse_pool.keys()):
            if self._reuse_pool[pid]["last_seen_frame"] < cutoff:
                del self._reuse_pool[pid]

        self._frame_count += 1

        # 排序: 关键点可见度高的排前面
        results.sort(key=lambda r: r.visible_keypoints, reverse=True)
        return results

    # ──────────────────────────────────────────────
    # 匹配
    # ──────────────────────────────────────────────
    def _match_detections(self, boxes_data, frame, h, w) -> Dict[Tuple, int]:
        if not boxes_data:
            return {}

        # 匹配池 = 活跃轨 + 近期短暂消失轨 (disappeared <= GRACE_FRAMES)
        pool = {pid: t for pid, t in self._tracked.items()
                if t["disappeared"] <= self.GRACE_FRAMES}

        # 先用联合打分做"主匹配" (针对池内出现过且未消失太久的轨)
        result = {}
        matched_det = set()
        matched_pool = set()
        if pool:
            # 计算每个检测与每个池轨的综合得分
            det_indices = list(range(len(boxes_data)))
            pool_ids = list(pool.keys())
            scores = np.full((len(boxes_data), len(pool_ids)), -1.0, dtype=np.float32)

            for i, det in enumerate(boxes_data):
                dx1, dy1, dx2, dy2, _ = det
                dcx = (dx1 + dx2) / 2.0
                dcy = (dy1 + dy2) / 2.0
                darea = (dx2 - dx1) * (dy2 - dy1)
                for j, pid in enumerate(pool_ids):
                    t = pool[pid]
                    # 优先用预测框(若有), 否则用最后 bbox
                    pbox = t.get("pred_box") if t.get("pred_box") else t["bbox"]
                    px1, py1, px2, py2 = pbox
                    iou = self._iou(dx1, dy1, dx2, dy2, px1, py1, px2, py2)

                    # 中心距离 (归一化到对角线)
                    pcx = (px1 + px2) / 2.0
                    pcy = (py1 + py2) / 2.0
                    diag = max(float(np.hypot(h, w)), 1.0)
                    center_dist = float(np.hypot(dcx - pcx, dcy - pcy) / diag)
                    center_sim = max(0.0, 1.0 - center_dist)

                    # 尺寸相似度
                    parea = (px2 - px1) * (py2 - py1)
                    size_sim = 0.0
                    if parea > 0 and darea > 0:
                        size_sim = min(darea, parea) / max(darea, parea)

                    # 外观相似度
                    app_sim = 0.0
                    ref_app = t.get("appearance")
                    det_app = _extract_appearance(frame, (dx1, dy1, dx2, dy2))
                    app_sim = _appearance_sim(ref_app, det_app)

                    # 联合得分
                    s = (self.W_IOU * iou
                         + self.W_CENTER * center_sim
                         + self.W_SIZE * size_sim
                         + self.W_APP * app_sim)
                    # 位置不合理(中心太远且 IoU 太小)则直接否决
                    if iou < 0.02 and center_dist > self.REID_MAX_CENTER:
                        s = -1.0
                    scores[i, j] = s

            # 贪心匹配联合得分
            for _ in range(min(len(boxes_data), len(pool_ids))):
                best_s = self.MATCH_JOINT_THRESH
                best_i, best_j = -1, -1
                for i in range(len(boxes_data)):
                    if i in matched_det:
                        continue
                    for j in range(len(pool_ids)):
                        if j in matched_pool:
                            continue
                        if scores[i, j] > best_s:
                            best_s = scores[i, j]
                            best_i, best_j = i, j
                if best_i >= 0:
                    matched_det.add(best_i)
                    matched_pool.add(best_j)
                    # 若该轨已从活跃转移到"即将消失", 这里重新激活
                    result[boxes_data[best_i]] = pool_ids[best_j]

        # ── 步骤 3: 未匹配的检测 → 软重识别 (search reuse pool + 活跃未匹配) ──
        unmatched_dets = [i for i in range(len(boxes_data)) if i not in matched_det]
        # 候选: 所有活跃轨(含消失中的) + reuse 池, 排除已匹配
        cand_pool = {}
        for pid, t in self._tracked.items():
            if pid not in result.values():
                cand_pool[pid] = {"appearance": t.get("appearance"),
                                  "bbox": t.get("last_bbox", t["bbox"]),
                                  "disappeared": t["disappeared"]}
        for pid, t in self._reuse_pool.items():
            if pid not in result.values():
                cand_pool[pid] = {"appearance": t["appearance"],
                                  "bbox": t["last_bbox"],
                                  "disappeared": 999}

        for i in unmatched_dets:
            det = boxes_data[i]
            dx1, dy1, dx2, dy2, _ = det
            dcx = (dx1 + dx2) / 2.0
            dcy = (dy1 + dy2) / 2.0
            diag = max(float(np.hypot(h, w)), 1.0)
            det_app = _extract_appearance(frame, (dx1, dy1, dx2, dy2))

            best_pid = None
            best_score = -1.0
            for pid, c in cand_pool.items():
                if c.get("appearance") is None:
                    continue
                pbox = c["bbox"]
                px1, py1, px2, py2 = pbox
                iou = self._iou(dx1, dy1, dx2, dy2, px1, py1, px2, py2)
                pcx = (px1 + px2) / 2.0
                pcy = (py1 + py2) / 2.0
                center_dist = float(np.hypot(dcx - pcx, dcy - pcy) / diag)
                app_sim = _appearance_sim(c["appearance"], det_app)
                # 组合: 外观为主, 位置作合理性约束
                score = 0.75 * app_sim + 0.25 * max(0.0, 1.0 - center_dist)
                if (app_sim >= self.REID_MIN_APP
                        and center_dist <= self.REID_MAX_CENTER
                        and iou >= self.REID_MIN_IOU
                        and score > best_score):
                    best_score = score
                    best_pid = pid
            if best_pid is not None:
                result[det] = best_pid
                # 从 reuse 池移回活跃(若在池里)
                if best_pid in self._reuse_pool:
                    del self._reuse_pool[best_pid]
                # 若最佳匹配是 reuse 池的旧 ID, 需要重新初始化活跃轨
                if best_pid not in self._tracked:
                    self._tracked[best_pid] = self._new_track(det[:4], frame)
                    self._tracked[best_pid]["appearance"] = cand_pool[best_pid]["appearance"]
                matched_det.add(i)

        # ── 步骤 4: 仍未被匹配的检测 → 分配新 ID ──
        for i in range(len(boxes_data)):
            if i in matched_det:
                continue
            det = boxes_data[i]
            pid = self._next_id
            self._next_id += 1
            self._tracked[pid] = self._new_track(det[:4], frame)
            result[det] = pid

        return result

    def _new_track(self, box, frame) -> Dict:
        """初始化一个新轨"""
        return {
            "bbox": box,
            "last_bbox": box,
            "pred_box": None,
            "disappeared": 0,
            "conf": 0.0,
            "kf": _BoxKF(box),
            "appearance": _extract_appearance(frame, box),
        }

    # ──────────────────────────────────────────────
    # MediaPipe 单人关键点
    # ──────────────────────────────────────────────
    def _pose_on_box(self, frame, box, h, w):
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 20 or box_h <= 40:
            return None, False, 0.0
        pad_x = int(box_w * 0.1)
        pad_y = int(box_h * 0.1)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None, False, 0.0
        t_mp = time.time()
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_results = self.pose.process(crop_rgb)
        mp_ms = (time.time() - t_mp) * 1000
        if mp_results.pose_landmarks:
            landmarks = np.zeros((33, 4), dtype=np.float32)
            ch, cw = crop.shape[:2]
            for i, lm in enumerate(mp_results.pose_landmarks.landmark):
                landmarks[i] = [
                    (lm.x * cw + cx1) / w,
                    (lm.y * ch + cy1) / h,
                    lm.z,
                    lm.visibility,
                ]
            return landmarks, True, mp_ms
        return None, False, mp_ms

    # ──────────────────────────────────────────────
    # 工具
    # ──────────────────────────────────────────────
    @staticmethod
    def _iou(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b) -> float:
        xi1 = max(x1a, x1b); yi1 = max(y1a, y1b)
        xi2 = min(x2a, x2b); yi2 = min(y2a, y2b)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area_a = (x2a - x1a) * (y2a - y1a)
        area_b = (x2b - x1b) * (y2b - y1b)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def get_status(self) -> dict:
        active = sum(1 for t in self._tracked.values() if t["disappeared"] <= 1)
        return {
            "total_detected": self._total_detected,
            "tracked_count": len(self._tracked),
            "active_count": active,
            "reuse_count": len(self._reuse_pool),
            "history_ids": len(self.id_frame_counts),
            "yolo_ms": round(self.yolo_infer_ms, 1),
        }

    def get_id_counts(self) -> Dict[int, int]:
        """返回 {person_id: 累计出现帧数}, 用于统计 ID 碎片化程度"""
        return dict(self.id_frame_counts)

    def close(self):
        try:
            self.pose.close()
        except (ValueError, AttributeError):
            pass


# 兼容别名: 若其他地方用 PersonTracker 引用, 可从这里取
PersonTracker = StablePersonTracker


# ════════════════════════════════════════════════════
# 测试入口 (headless, 无需 GUI)
# ════════════════════════════════════════════════════
def main():
    import argparse
    ap = argparse.ArgumentParser(description="StablePersonTracker headless 测试")
    ap.add_argument("--source", "-s", type=str, required=True, help="视频文件路径")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--max-frames", type=int, default=0, help="最多处理帧数(0=全部)")
    ap.add_argument("--frame-step", type=int, default=1, help="隔几帧处理一帧")
    ap.add_argument("--min-active-frames", type=int, default=10,
                    help="统计活跃ID的最少帧数")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("[ERROR] 无法打开视频:", args.source)
        return
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[setup] 视频 {cap_w}x{cap_h}, 总帧 {total_frames}")

    tracker = StablePersonTracker(confidence=args.conf)

    frame_idx = 0
    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and processed >= args.max_frames:
            break
        if frame_idx % args.frame_step != 0:
            frame_idx += 1
            continue

        tracker.update(frame)
        processed += 1
        if processed % 50 == 0:
            st = tracker.get_status()
            print(f"  [proc {processed}/{total_frames}] "
                  f"active={st['active_count']} tracked={st['tracked_count']} "
                  f"reuse={st['reuse_count']} hist_ids={st['history_ids']}")
        frame_idx += 1

    cap.release()
    tracker.close()

    # ── 汇总统计 ──
    counts = tracker.get_id_counts()
    print("\n===== ID 碎片化统计 =====")
    total_ids = len(counts)
    active_ids = sum(1 for c in counts.values() if c >= args.min_active_frames)
    print(f"总生成 ID 数: {total_ids}")
    print(f"活跃 ID 数(出现>={args.min_active_frames}帧): {active_ids}")
    if counts:
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
        print("Top 持续 ID (person_id: 帧数):")
        for pid, cnt in top:
            print(f"  P{pid}: {cnt} 帧")
    print(f"历史最大同时活跃: {tracker.get_status()['total_detected']}")
    return total_ids, active_ids


if __name__ == "__main__":
    main()
