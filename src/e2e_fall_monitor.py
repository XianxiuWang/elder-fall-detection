#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e2e_fall_monitor.py — 端到端跌倒监控（PC 端全链路验证）
=====================================================
v6.2/v6.3 更新 — 踉跄前兆 + 行为画像 + 暖关怀:
  - v6.2 踉跄前兆状态机: 复用 early_warning 6指标 → 离散状态机
    IDLE→WATCH→STUMBLING→DANGER→FALL_DETECTED, DANGER触发"你还好吗?"暖语音
  - v6.3 行为画像: 把 ML类别流变成长期行为时间线, "今天和平时有什么不同"
  - 均为每人独立 (挂 PersonState), 无新模型, 可解释

v6.1 更新 — 个性化步态基线:
  - 每人独立的步态"指纹" (7项核心指标 × μ/σ)
  - Z-score 偏离检测 (和"过去的自己"比, 而非人群平均)
  - 基线自动校准 + EMA 慢速自适应 + 跨会话持久化
  - 多人面板显示个人基线偏离信息

v6.0 更新 — 多人检测支持:
  - YOLOv8n 行人检测 + IoU 跟踪 → 每人独立 ID
  - 逐人 MediaPipe Pose 33 关键点
  - 每人均运行完整检测管线 (ML六分类 + 跌倒预测 + 早期预警)
  - 任意一人出现跌倒风险 → 全局预警 + 告警
  - 每人不同颜色骨架 + 独立信息面板

管线 (单人模式):
  RTSP/摄像头 → MediaPipe → 33关键点 → ML六分类 + 空间运动 + 过程检测
                                  → 步态趋势 + 久坐 + 服药提醒
                                  → 跌倒预测 + 早期预警 → 告警
                                  → 踉跄前兆状态机 → 暖关怀语音
                                  → 行为画像 → "今天和平时有什么不同"

管线 (多人模式 --multi):
  RTSP/摄像头 → YOLOv8n 行人检测 → 逐人 MediaPipe → 各自独立管线
                                  → 任一风险 → 全局告警

用法:
    conda activate fall
    python -m src.e2e_fall_monitor                                       # 单人(默认)
    python -m src.e2e_fall_monitor --multi                               # 多人模式
    python -m src.e2e_fall_monitor --source "video.mp4"                  # 视频文件
    python -m src.e2e_fall_monitor --source "video.mp4" --multi          # 视频+多人
"""

import cv2
import numpy as np
import time
import sys
import os
import argparse

sys.path.insert(0, r'E:\老人跌倒')

import mediapipe as mp

from .motion_spatial import MotionSpatialAnalyzer, MotionSpatial
from .process_fall_detector import ProcessFallDetector, FallAlert
from .gait_trend import GaitTrendAnalyzer, GaitSample, GaitMetricExtractor
from .personalized_baseline import PersonalBaseline, BaselineConfig
from .fall_config import FallConfig, load_or_default
from .ml_6class_detector import ML6ClassDetector
from .sedentary_detector import SedentaryDetector
from .medication_reminder import MedicationReminder
from .fall_predictor import FallPredictor
from .fall_early_warning import FallEarlyWarning
from .activity_logger import ActivityLogger
from .alert_manager import AlertManager
# v6.2+ 创新模块: 踉跄前兆状态机 + 暖关怀语音 + 行为画像
from .stumble_precursor import StumblePrecursorAnalyzer, CareVoice
from .behavior_profile import BehaviorProfile

# v6.0: 多人检测支持
try:
    from .multi_person import PersonTracker, PersonResult
    _MULTI_PERSON_AVAILABLE = True
except ImportError as e:
    _MULTI_PERSON_AVAILABLE = False
    PersonTracker = None
    PersonResult = None

# 默认模型路径
_DEFAULT_MODEL = r"E:\老人跌倒\models\fall_classifier_6class.pkl"

# 每人分配的颜色 (最多 6 人)
_PERSON_COLORS = [
    (0, 255, 0),      # 绿色
    (0, 165, 255),    # 橙色
    (255, 255, 0),    # 青色
    (255, 0, 255),    # 品红
    (128, 255, 0),    # 黄绿
    (0, 255, 255),    # 黄色
]


# ════════════════════════════════════════════════
# 每人独立检测器状态
# ════════════════════════════════════════════════

class PersonState:
    """封装单个行人的全部检测器状态"""

    __slots__ = (
        "person_id", "ml_detector", "predictor", "early_warning",
        "motion_analyzer", "fall_detector", "gait_analyzer",
        "sedentary", "medication", "prev_lm", "color",
        "total_frames", "lost_frames", "last_seen",
        "last_alert",
        "alert_frames", "alert_total",
        "baseline", "metric_extractor",  # v6.1: 个性化基线
        "care_voice", "stumble", "behavior",   # v6.2+: 踉跄状态机 + 暖语音 + 行为画像
        "stumble_state", "stumble_risk", "behavior_insight",  # 缓存反馈HUD
    )

    def __init__(self, person_id: int, model_path: str, fps: float, config: FallConfig):
        self.person_id = person_id
        self.prev_lm = None
        self.total_frames = 0
        self.lost_frames = 0
        self.last_seen = 0           # v6.1: 最后出现的帧号 (用于清理)
        self.last_alert = None
        self.alert_frames = 0
        self.alert_total = 0
        self.color = _PERSON_COLORS[person_id % len(_PERSON_COLORS)]

        # ML 六分类器
        if os.path.exists(model_path):
            try:
                self.ml_detector = ML6ClassDetector(
                    model_path, window_size=30, stride=5, fall_threshold=0.6)
            except Exception as e:
                print(f"  [Person {person_id}] ML 加载失败: {e}, 回退启发式")
                self.ml_detector = None
        else:
            self.ml_detector = None

        # 其他检测器
        self.motion_analyzer = MotionSpatialAnalyzer(config=config.motion_spatial)
        self.fall_detector = ProcessFallDetector(config=config.process_fall)
        self.gait_analyzer = GaitTrendAnalyzer(config=config.gait_trend)
        self.predictor = FallPredictor(fps=fps)
        self.early_warning = FallEarlyWarning(fps=fps)
        self.sedentary = SedentaryDetector(
            alert_minutes=config.e2e_monitor.sedentary_alert_min,
            warn_minutes=config.e2e_monitor.sedentary_warn_min,
        )
        self.medication = MedicationReminder()

        # v6.1: 个性化步态基线 (每人独立)
        self.metric_extractor = GaitMetricExtractor(config=config.gait_trend)
        self.baseline = PersonalBaseline(person_id=f"P{person_id}")

        # v6.2+: 踉跄前兆状态机 + 暖关怀语音 (方向二)
        self.care_voice = CareVoice(language="zh-CN")
        self.stumble = StumblePrecursorAnalyzer(
            early_warn=self.early_warning,
            voice=self.care_voice,
            person_id=f"P{person_id}",
            fps=fps)
        # v6.3: 行为画像 (方向三)
        self.behavior = BehaviorProfile(person_id=f"P{person_id}", fps=fps)
        # HUD 缓存
        self.stumble_state = "IDLE"
        self.stumble_risk = 0.0
        self.behavior_insight = None


# ════════════════════════════════════════════════
# 管线主类
# ════════════════════════════════════════════════

class FallMonitorPipeline:
    """封装完整管线 (v6.0: 支持多人检测)"""

    def __init__(self, config: FallConfig, model_path: str = _DEFAULT_MODEL,
                 use_ml: bool = True, source: str = None,
                 use_multi_person: bool = False):
        self.cfg = config
        self.frame_idx = 0
        self.use_ml = use_ml and os.path.exists(model_path)
        self.source = source
        self.use_multi_person = use_multi_person

        if use_ml and not self.use_ml:
            print(f"  [WARN] ML 模型未找到: {model_path}, 回退启发式")

        # ── 初始化摄像头 / 视频文件 ──
        self._init_camera()

        # ── 初始化检测器 ──
        if self.use_multi_person and _MULTI_PERSON_AVAILABLE:
            self._init_multi_person(model_path)
        else:
            if self.use_multi_person and not _MULTI_PERSON_AVAILABLE:
                print("  [WARN] 多人检测模块不可用, 回退单人模式")
                self.use_multi_person = False
            self._init_single_person(model_path)

    # ── 摄像头/视频初始化 (同 v5.x) ──

    def _init_camera(self):
        cfg = self.cfg.e2e_monitor
        if self.source:
            print(f"[1/5] 加载视频文件: {self.source}")
            self.cap = cv2.VideoCapture(self.source)
            self._is_video_file = True
        else:
            print(f"[1/5] 连接摄像头: {cfg.rtsp_url[:50]}...")
            self.cap = cv2.VideoCapture(cfg.rtsp_url)
            self._is_video_file = False
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.camera_buffer_size)

        if not self.cap.isOpened():
            if cfg.use_local_camera_fallback:
                print("  [WARN] RTSP 失败，回退到本地摄像头...")
                self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("无法打开任何摄像头")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self._is_video_file and src_fps > 0:
            self.fps = src_fps
        elif self.fps <= 0:
            self.fps = 30.0
        src_label = f"视频: {os.path.basename(self.source)}" if self._is_video_file else "摄像头"
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_info = f", {total_frames}帧" if self._is_video_file and total_frames > 0 else ""
        print(f"  [OK] {src_label} {self.width}x{self.height} @ {self.fps:.0f}fps{frame_info}")

    # ── 单人模式初始化 (原 v5.x 逻辑) ──

    def _init_single_person(self, model_path: str):
        mp_cfg = self.cfg.mediapipe
        print(f"[2/5] 初始化 MediaPipe (单人, complexity={mp_cfg.model_complexity})...")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=mp_cfg.static_image_mode,
            model_complexity=mp_cfg.model_complexity,
            smooth_landmarks=mp_cfg.smooth_landmarks,
            min_detection_confidence=mp_cfg.min_detection_confidence,
            min_tracking_confidence=mp_cfg.min_tracking_confidence,
        )
        print("  [OK]")

        print(f"[3/5] 初始化检测模块...")
        self.motion_analyzer = MotionSpatialAnalyzer(config=self.cfg.motion_spatial)
        self.fall_detector = ProcessFallDetector(config=self.cfg.process_fall)
        self.gait_analyzer = GaitTrendAnalyzer(config=self.cfg.gait_trend)
        if self.use_ml:
            self.ml_detector = ML6ClassDetector(model_path, window_size=30, stride=5, fall_threshold=0.6)
        else:
            self.ml_detector = None
        self.sedentary = SedentaryDetector(
            alert_minutes=self.cfg.e2e_monitor.sedentary_alert_min,
            warn_minutes=self.cfg.e2e_monitor.sedentary_warn_min,
        )
        self.medication = MedicationReminder()
        self.predictor = FallPredictor(fps=self.fps)
        self.early_warning = FallEarlyWarning(fps=self.fps)
        self.logger = ActivityLogger()
        self.logger.start_session(fps=self.fps)
        self.alerts = AlertManager(enable_audio=True, enable_screenshots=True)
        self.prev_lm = None
        self.lost_frames = 0
        self.last_alert = None
        self.alert_display_frames = 0
        self.alert_display_total = 0
        self._last_early_warn = None
        # v6.1: 单人模式也加个性化基线
        self._single_baseline = PersonalBaseline(person_id="P0")
        self._single_metric_ext = GaitMetricExtractor(config=self.cfg.gait_trend)
        print(f"  [OK] motion_spatial + process_fall + gait_trend"
              f"{' + ML_6CLASS' if self.ml_detector else ''}"
              f" + sedentary + medication + predict + early_warning")

    # ── 多人模式初始化 ──

    def _init_multi_person(self, model_path: str):
        print(f"[2/5] 初始化多人检测 (YOLOv8n + MediaPipe)...")
        self.person_tracker = PersonTracker(confidence=0.45)
        print("  [OK]")

        print(f"[3/5] 初始化检测模块 (全局共享)...")
        self.logger = ActivityLogger()
        self.logger.start_session(fps=self.fps)
        self.alerts = AlertManager(enable_audio=True, enable_screenshots=True)
        self._person_states: dict = {}      # {person_id: PersonState}
        self._model_path = model_path
        self._last_early_warn = None
        self._first_person_created = False  # v6.1: 锁, 只打印一次模型加载信息
        print(f"  [OK] 每人独立检测器延迟初始化, 模型: {model_path}")

    def _get_or_create_person_state(self, person_id: int) -> PersonState:
        """延迟创建每人独立的检测器 (v6.1: 静默创建, 仅首次打印模型信息)"""
        if person_id not in self._person_states:
            state = PersonState(person_id, self._model_path, self.fps, self.cfg)
            self._person_states[person_id] = state
            if not self._first_person_created:
                self._first_person_created = True
                info = f"模型已加载(每人独立): {os.path.basename(self._model_path)}"
                print(f"  [OK] {info}")
        return self._person_states[person_id]

    # ════════════════════════════════════════════════
    # 多人模式: 处理单个人
    # ════════════════════════════════════════════════

    def _process_person(self, lm: np.ndarray, state: PersonState, frame) -> dict:
        """对一个人的关键点跑完整检测管线, 返回结果字典"""
        h, w = frame.shape[:2]
        result = {
            "person_id": state.person_id,
            "ml_result": None,
            "pred_report": None,
            "early_warn": None,
            "alert_triggered": False,
            "spatial": None,
            "fall_alert": None,
            "sedentary_msg": "",
            "med_msg": "",
            "baseline_report": None,  # v6.1: 个性化基线报告
            "stumble_event": None,    # v6.2: 踉跄前兆事件
            "stumble_state": "IDLE",  # v6.2: 状态缓存
            "stumble_risk": 0.0,      # v6.2: 连续风险
            "behavior_summary": None, # v6.3: 行为画像摘要
        }

        # 空间运动
        spatial = state.motion_analyzer.extract(lm, state.prev_lm)
        spatial_arr = spatial.to_array()
        result["spatial"] = spatial

        # ML 六分类
        if state.ml_detector:
            ml_result = state.ml_detector.update(lm)
            result["ml_result"] = ml_result
        else:
            ml_result = None

        # 启发式过程检测
        feat_vec = np.zeros(100, dtype=np.float32)
        feat_vec[1] = lm[23:25, 1].mean()
        feat_vec[3] = self._compute_torso_angle(lm)
        feat_vec[5] = lm[11:13, 1].mean()
        alert = state.fall_detector.update(feat_vec, spatial_arr, state.total_frames)
        if alert is not None:
            state.last_alert = alert
            state.alert_total = int(self.fps * self.cfg.e2e_monitor.alert_display_seconds)
            state.alert_frames = state.alert_total
            result["fall_alert"] = alert

        # ML 触发摔倒告警 (v6.1: 需要足够预热帧数, 避免冷启动假阳性)
        _WARMUP_FRAMES = 45
        if ml_result and ml_result.fall_triggered and ml_result.inference_count >= _WARMUP_FRAMES:
            state.alert_total = int(self.fps * self.cfg.e2e_monitor.alert_display_seconds)
            state.alert_frames = state.alert_total
            result["alert_triggered"] = True

        # 久坐
        _, sed_msg = state.sedentary.update(lm)
        result["sedentary_msg"] = sed_msg

        # 服药
        _, med_msg = state.medication.update(lm)
        result["med_msg"] = med_msg

        # 跌倒预测
        pred_report = state.predictor.update(lm)
        result["pred_report"] = pred_report

        # 早期预警
        is_walking = state.predictor._is_walking if hasattr(state.predictor, '_is_walking') else True
        early_warn = state.early_warning.update(
            lm, elapsed=state.total_frames / self.fps, is_walking=is_walking)
        result["early_warn"] = early_warn

        # 步态
        state.gait_analyzer.update(features=feat_vec, landmarks=lm,
                                    prev_landmarks=state.prev_lm)

        # v6.1: 个性化基线 — 和"过去的自己"比
        try:
            gait_sample = state.metric_extractor.extract(
                feat_vec, landmarks=lm, prev_landmarks=state.prev_lm)
            gait_sample.is_walking = (
                state.predictor._is_walking
                if hasattr(state.predictor, '_is_walking') else True
            )
            baseline_report = state.baseline.update(gait_sample)
            if baseline_report is not None:
                result["baseline_report"] = baseline_report
        except Exception:
            pass  # 基线不影响主流程

        # v6.2+: 踉跄前兆状态机 (方向二) — 复用early_warning, 前置2-5秒
        try:
            stumble_evt = state.stumble.update(
                lm, ml_result, self.frame_idx,
                elapsed=state.total_frames / self.fps)
            state.stumble_state = state.stumble.state
            state.stumble_risk = state.stumble.get_status()["recent_risk"]
            result["stumble_event"] = stumble_evt
            result["stumble_state"] = state.stumble_state
            result["stumble_risk"] = state.stumble_risk
            # 踉跄触发DANGER → 暖关怀语音已在状态机内播放
            if stumble_evt and stumble_evt.state == "DANGER":
                result["stumble_triggered"] = True
                if stumble_evt.direction_hint:
                    result["stumble_direction"] = stumble_evt.direction_hint
        except Exception as e:
            result["stumble_event"] = None
            state.stumble_state = "ERR"

        # v6.3: 行为画像 (方向三) — 记录ML类别到长期时间线
        try:
            if ml_result is not None and ml_result.class_id >= 0:
                state.behavior.update(ml_result.class_id)
                # 关联风险事件到画像 (基线ALERT / 踉跄DANGER)
                br_ = result.get("baseline_report")
                if br_ and br_.alert_level == "ALERT":
                    state.behavior.register_risk_event("ALERT")
                if result.get("stumble_triggered"):
                    state.behavior.register_risk_event("DANGER")
                # 每日洞察低频刷新
                if state.total_frames % 300 == 0 or state.behavior_insight is None:
                    state.behavior_insight = state.behavior.report_today()
                # 摘要: 今天vs平时 (简短, 适配HUD)
                bi = state.behavior_insight
                if bi is not None and bi.has_enough_data and bi.walk_deviation_pct is not None and abs(bi.walk_deviation_pct) > 10:
                    result["behavior_summary"] = f"walk{bi.walk_deviation_pct:+.0f}%"
        except Exception:
            pass  # 画像不影响主流程

        # 告警通知
        if pred_report:
            self.alerts.update(
                risk_score=pred_report.risk_score,
                alert_level=pred_report.alert_level,
                ml_fall_triggered=ml_result.fall_triggered if ml_result else False,
                fall_direction=pred_report.fall_direction if pred_report else "none",
                frame=frame,
            )

        # 日志
        self.logger.log_frame(
            risk_report=pred_report,
            ml_result=ml_result,
            frame_idx=self.frame_idx,
            timestamp=time.time(),
            alert_active=(state.alert_frames > 0),
        )

        # 更新状态
        state.prev_lm = lm
        state.lost_frames = 0
        state.total_frames += 1
        state.last_seen = self.frame_idx  # v6.1: 跟踪最后出现帧
        if state.alert_frames > 0:
            state.alert_frames -= 1

        return result

    # ════════════════════════════════════════════════
    # 多人模式: 处理帧
    # ════════════════════════════════════════════════

    def process_frame(self, frame) -> np.ndarray:
        """处理一帧 (v6.0: 支持多人)"""
        if self.use_multi_person and _MULTI_PERSON_AVAILABLE:
            return self._process_frame_multi(frame)
        else:
            return self._process_frame_single(frame)

    def _process_frame_multi(self, frame) -> np.ndarray:
        """多人模式帧处理"""
        h, w = frame.shape[:2]

        # 1. 检测所有行人
        persons = self.person_tracker.update(frame)

        if not persons:
            cv2.putText(frame, "No person detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.cfg.e2e_monitor.gray, 2)
            self.frame_idx += 1
            return frame

        # 2. 处理每个人
        all_results = []
        any_fall = False
        highest_risk_person_id = -1
        highest_risk = 0.0

        for p in persons:
            if not p.has_landmarks or p.landmarks is None:
                # 画出边界框但关键点不可用→跳过深度分析
                x1, y1, x2, y2 = p.bbox
                color = _PERSON_COLORS[p.person_id % len(_PERSON_COLORS)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(frame, f"ID:{p.person_id} (no kp)", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                continue

            # 获取此人的检测器状态
            state = self._get_or_create_person_state(p.person_id)
            lm = p.landmarks

            # 跑完整管线
            result = self._process_person(lm, state, frame)

            # 画此人的骨架和边界框
            self._draw_person_skeleton(frame, lm, state.color)
            self._draw_person_label(frame, p, state, result)
            all_results.append(result)

            # 追踪风险
            if result.get("alert_triggered"):
                any_fall = True
            if result.get("pred_report"):
                risk = result["pred_report"].risk_score
                if risk > highest_risk:
                    highest_risk = risk
                    highest_risk_person_id = p.person_id

        # 3. 画多人信息面板 + 全局告警
        self._draw_multi_panel(frame, all_results, persons)

        # 4. 全局告警 (任一人触发)
        global_alert = any_fall or (highest_risk >= 60 and len(persons) >= 2)
        if global_alert:
            self._draw_global_alert(frame, highest_risk, highest_risk_person_id, all_results)

        # 5. 帧信息
        status = self.person_tracker.get_status()
        cv2.putText(frame,
                    f"F:{self.frame_idx} | YOLO:{status['yolo_ms']:.0f}ms "
                    f"| Persons:{len(persons)}({status['active_count']} active)",
                    (w - 500, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 6. 进度日志 (v6.1: 精简摘要, 仅突出风险)
        if self.frame_idx % self.cfg.e2e_monitor.progress_log_interval == 0:
            n_persons = len(persons)
            # 只显示有风险的人
            risks = []
            for r in all_results:
                pid = r["person_id"]
                pred = r.get("pred_report")
                br = r.get("baseline_report")
                if pred and pred.risk_score > 30:
                    risks.append(f"P{pid}_risk:{pred.risk_score:.0f}")
                if br and br.calibrated and br.personal_risk_score > 15:
                    risks.append(f"P{pid}_base:{br.personal_risk_score:.0f}")
            if risks:
                print(f"  [{self.frame_idx:>4d}f | {n_persons}人] ⚠ {' '.join(risks)}")
            else:
                print(f"  [{self.frame_idx:>4d}f | {n_persons}人] OK")

            # v6.1: 清理僵尸人物状态 (超过600帧未出现 = 20秒)
            stale_ids = [pid for pid, s in self._person_states.items()
                         if self.frame_idx - s.last_seen > 600]
            for pid in stale_ids:
                # 如果基线已校准, 最后保存一次
                state = self._person_states[pid]
                if state.total_frames > 30:  # 至少有点数据才保存
                    try:
                        state.baseline.save()
                    except Exception:
                        pass
                # 释放资源
                try:
                    if state.ml_detector:
                        del state.ml_detector
                except Exception:
                    pass
                del self._person_states[pid]
            if stale_ids:
                print(f"  [GC] 清理 {len(stale_ids)} 个长时间未出现的人物")

        self.frame_idx += 1
        return frame

    # ════════════════════════════════════════════════
    # 单人模式: 处理帧 (原 v5.x 逻辑, 保持不变)
    # ════════════════════════════════════════════════

    def _process_frame_single(self, frame) -> np.ndarray:
        """单人模式帧处理 (v5.x 原版, 仅在非多人模式下使用)"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        h, w = frame.shape[:2]

        if results.pose_landmarks:
            lm = self._extract_landmarks(results)
            self._draw_skeleton(frame, results)

            spatial = self.motion_analyzer.extract(lm, self.prev_lm)
            spatial_arr = spatial.to_array()

            if self.ml_detector:
                ml_result = self.ml_detector.update(lm)
            else:
                ml_result = None

            feat_vec = np.zeros(100, dtype=np.float32)
            feat_vec[1] = lm[23:25, 1].mean()
            feat_vec[3] = self._compute_torso_angle(lm)
            feat_vec[5] = lm[11:13, 1].mean()
            alert = self.fall_detector.update(feat_vec, spatial_arr, self.frame_idx)
            if alert is not None:
                self.last_alert = alert
                self.alert_display_total = int(self.fps * self.cfg.e2e_monitor.alert_display_seconds)
                self.alert_display_frames = self.alert_display_total

            if ml_result and ml_result.fall_triggered:
                self.alert_display_total = int(self.fps * self.cfg.e2e_monitor.alert_display_seconds)
                self.alert_display_frames = self.alert_display_total

            sedentary_sitting, sedentary_msg = self.sedentary.update(lm)
            med_alert, med_msg = self.medication.update(lm)
            pred_report = self.predictor.update(lm)

            is_walking = self.predictor._is_walking
            early_warn = self.early_warning.update(
                lm, elapsed=self.frame_idx / self.fps, is_walking=is_walking)
            self._last_early_warn = early_warn

            self.gait_analyzer.update(features=feat_vec, landmarks=lm,
                                      prev_landmarks=self.prev_lm)

            # v6.1: 个性化基线 (单人模式)
            try:
                gait_sample = self._single_metric_ext.extract(
                    feat_vec, landmarks=lm, prev_landmarks=self.prev_lm)
                gait_sample.is_walking = is_walking
                self._single_baseline.update(gait_sample)
            except Exception:
                pass

            self._draw_hud_v4(frame, spatial, lm, ml_result, pred_report,
                              early_warn, sedentary_sitting, med_msg)

            self.logger.log_frame(
                risk_report=pred_report,
                ml_result=ml_result,
                frame_idx=self.frame_idx,
                timestamp=time.time(),
                alert_active=(self.alert_display_frames > 0),
            )

            self.alerts.update(
                risk_score=pred_report.risk_score if pred_report else 0,
                alert_level=pred_report.alert_level if pred_report else "SAFE",
                ml_fall_triggered=ml_result.fall_triggered if ml_result else False,
                fall_direction=pred_report.fall_direction if pred_report else "none",
                frame=frame,
            )

            self.prev_lm = lm
            self.lost_frames = 0
        else:
            self.lost_frames += 1
            frame = self._handle_lost_tracking(frame)

        if self.alert_display_frames > 0:
            self.alert_display_frames -= 1

        self.frame_idx += 1
        if self.frame_idx % self.cfg.e2e_monitor.progress_log_interval == 0:
            self._log_single_progress()

        return frame

    def _handle_lost_tracking(self, frame):
        """单人模式: 处理跟踪丢失"""
        if self.prev_lm is not None and self.lost_frames <= 10:
            self._draw_low_confidence_skeleton(frame)
            lm = self.prev_lm
            if self.ml_detector:
                self.ml_detector.update(lm)
            self.sedentary.update(lm)
            self.medication.update(lm)
            self.predictor.update(lm)
            pred_report = self.predictor.last_report
            self.early_warning.update(lm, elapsed=self.frame_idx / self.fps,
                                      is_walking=self.predictor._is_walking)
            feat_vec = np.zeros(100, dtype=np.float32)
            feat_vec[1] = lm[23:25, 1].mean()
            feat_vec[3] = self._compute_torso_angle(lm)
            feat_vec[5] = lm[11:13, 1].mean()
            spatial = self.motion_analyzer.extract(lm, self.prev_lm)
            spatial_arr = spatial.to_array()
            self.fall_detector.update(feat_vec, spatial_arr, self.frame_idx)
            self.gait_analyzer.update(features=feat_vec, landmarks=lm,
                                      prev_landmarks=self.prev_lm)
            # v6.1: 个性化基线 (丢失跟踪时仍更新)
            try:
                gait_sample = self._single_metric_ext.extract(
                    feat_vec, landmarks=lm, prev_landmarks=self.prev_lm)
                gait_sample.is_walking = self.predictor._is_walking
                self._single_baseline.update(gait_sample)
            except Exception:
                pass
            self._draw_hud_v4(frame, spatial, lm, None, pred_report, None, False, "")
        else:
            cv2.putText(frame, "No person detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.cfg.e2e_monitor.gray, 2)
        return frame

    def _log_single_progress(self):
        """单人模式: 进度日志"""
        stats_str = ""
        if self.ml_detector:
            s = self.ml_detector.get_stats()
            stats_str += f" | 动作: {s['stable_class_cn']}"
        pred_s = self.predictor.get_status()
        if pred_s['risk_score'] > 0:
            stats_str += f" | 风险: {pred_s['risk_score']:.0f} [{pred_s['alert_level']}]"
        walking = "行走" if pred_s.get('is_walking') else "静止"
        stats_str += f" | 状态: {walking}"
        ew = self._last_early_warn if hasattr(self, '_last_early_warn') else None
        if ew and ew.pre_fall_risk >= 30:
            stats_str += f" | [预测: {ew.pre_fall_risk:.0f} {ew.alert_level_cn}]"
        print(f"  [INFO] {self.frame_idx} 帧{stats_str}...")

    # ════════════════════════════════════════════════
    # 绘制：多人模式
    # ════════════════════════════════════════════════

    def _draw_person_skeleton(self, frame, lm: np.ndarray, color: tuple):
        """绘制单人骨架 (简化版，使用颜色区分)"""
        h, w = frame.shape[:2]
        # 关键点
        for i in range(33):
            if lm[i, 3] > 0.5:
                px, py = int(lm[i, 0] * w), int(lm[i, 1] * h)
                cv2.circle(frame, (px, py), 2, color, -1)

        # 连接线
        connections = [
            # 躯干
            (11, 12), (11, 23), (12, 24), (23, 24),
            # 左臂
            (11, 13), (13, 15),
            # 右臂
            (12, 14), (14, 16),
            # 左腿
            (23, 25), (25, 27),
            # 右腿
            (24, 26), (26, 28),
        ]
        for a, b in connections:
            if lm[a, 3] > 0.5 and lm[b, 3] > 0.5:
                p1 = (int(lm[a, 0] * w), int(lm[a, 1] * h))
                p2 = (int(lm[b, 0] * w), int(lm[b, 1] * h))
                cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)

    def _draw_person_label(self, frame, person: PersonResult,
                            state: PersonState, result: dict):
        """在每人 bbox 上标注状态标签"""
        x1, y1, x2, y2 = person.bbox
        color = state.color

        # 边界框
        thickness = 3 if result.get("alert_triggered") else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # 标签行
        ml = result.get("ml_result")
        pred = result.get("pred_report")
        ew = result.get("early_warn")

        label_parts = [f"ID:{person.person_id}"]

        if ml and ml.inference_done:
            label_parts.append(ml.class_name_cn)

        if pred:
            risk = pred.risk_score
            if risk >= 70:
                label_parts.append(f"!!RISK:{risk:.0f}")
            elif risk >= 50:
                label_parts.append(f"!RISK:{risk:.0f}")
            elif risk >= 30:
                label_parts.append(f"risk:{risk:.0f}")

        if ew and ew.pre_fall_risk >= 50:
            label_parts.append(f"PRE:{ew.pre_fall_risk:.0f}")

        label = " | ".join(label_parts)

        # 背景
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1 - 2),
                      (0, 0, 0), -1)
        # 告警闪烁效果
        if state.alert_frames > 0 and state.alert_frames % 10 < 5:
            text_color = (0, 0, 255)  # 红色闪烁
        else:
            text_color = color
        cv2.putText(frame, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

    def _draw_multi_panel(self, frame, all_results: list, persons: list):
        """多人信息面板 (右上角, 紧凑型)"""
        if not all_results:
            return

        h, w = frame.shape[:2]
        panel_x = w - 280
        panel_y = 5
        panel_w = 275
        panel_h = min(25 + len(all_results) * 105, h - 20)

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        y = panel_y + 18
        cv2.putText(frame, f"Persons: {len(persons)}", (panel_x + 5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 20

        for result in all_results:
            pid = result["person_id"]
            ml = result.get("ml_result")
            pred = result.get("pred_report")
            ew = result.get("early_warn")

            # 颜色标记
            color = _PERSON_COLORS[pid % len(_PERSON_COLORS)]
            cv2.putText(frame, f"P{pid}:", (panel_x + 5, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # ML 类别
            if ml and ml.inference_done:
                ml_label = ml.class_name_cn
                ml_color = (0, 0, 255) if ml.class_id == 0 else (0, 255, 0)
                cv2.putText(frame, f"{ml_label}", (panel_x + 45, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, ml_color, 1)
            y += 16

            # 风险分
            if pred:
                risk = pred.risk_score
                risk_color = ((0, 0, 255) if risk >= 70 else
                              (0, 140, 255) if risk >= 50 else
                              (0, 220, 255) if risk >= 30 else
                              (0, 255, 0))
                level = pred.alert_level
                cv2.putText(frame, f"  Risk:{risk:.0f} [{level}]",
                            (panel_x + 5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, risk_color, 1)
                y += 14

            # 早期预警
            if ew and ew.pre_fall_risk >= 30:
                ew_color = ((0, 0, 255) if ew.pre_fall_risk >= 70 else
                            (0, 140, 255) if ew.pre_fall_risk >= 50 else
                            (0, 220, 255))
                cv2.putText(frame, f"  Pre:{ew.pre_fall_risk:.0f} [{ew.alert_level_cn}]",
                            (panel_x + 5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, ew_color, 1)
                y += 14

            # v6.1: 个性化基线偏离
            br = result.get("baseline_report")
            if br is not None and br.calibrated and br.personal_risk_score > 15:
                br_color = ((0, 0, 255) if br.alert_level in ("DANGER", "ALERT") else
                            (0, 200, 255) if br.alert_level == "WARN" else
                            (255, 255, 255))
                cv2.putText(frame, f"  Base:{br.personal_risk_score:.0f} [{br.alert_level}]",
                            (panel_x + 5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, br_color, 1)
                y += 12

            # v6.2+: 踉跄前兆状态 (方向二)
            sd = result.get("stumble_event")
            st_state = result.get("stumble_state") or (sd.state if sd else "IDLE")
            if st_state and st_state not in ("IDLE", "ERR"):
                st_color = {
                    "WATCH": (0, 220, 255), "STUMBLING": (0, 140, 255),
                    "DANGER": (0, 0, 255), "FALL_DETECTED": (0, 0, 255),
                }.get(st_state, (255, 255, 255))
                st_risk = sd.pre_fall_risk if sd else 0.0
                cv2.putText(frame, f"  Stumble:{st_state} {st_risk:.0f}",
                            (panel_x + 5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.33, st_color, 1)
                y += 12

            # v6.3: 行为画像摘要 (方向三)
            bi = result.get("behavior_summary")
            if bi:
                cv2.putText(frame, f"  Behave:{bi}", (panel_x + 5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 255, 180), 1)
                y += 12

            # 分隔线
            y += 4

    def _draw_global_alert(self, frame, highest_risk: float,
                            highest_risk_id: int, all_results: list):
        """多人模式：全局告警横幅"""
        h, w = frame.shape[:2]
        c = self.cfg.e2e_monitor

        # 顶部横幅
        alert_y = 5
        cv2.rectangle(frame, (0, alert_y), (w, 45), (0, 0, 120), -1)

        if highest_risk >= 70:
            alert_text = f"!!! FALL DETECTED — Person {highest_risk_id} Risk:{highest_risk:.0f}/100 !!!"
            alert_color = (0, 0, 255)
        else:
            alert_text = f"!! FALL RISK WARNING — Person {highest_risk_id} Risk:{highest_risk:.0f}/100 !!"
            alert_color = (0, 165, 255)

        (ta_w, ta_h), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        tx = (w - ta_w) // 2
        cv2.putText(frame, alert_text, (tx, alert_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, alert_color, 2)

        # 底部告警栏
        bottom_y = h - 45
        cv2.rectangle(frame, (0, bottom_y), (w, h), (0, 0, 80), -1)

        # 显示每个人状态摘要
        parts = []
        for r in all_results:
            pid = r["person_id"]
            ml = r.get("ml_result")
            pred = r.get("pred_report")
            if ml:
                if ml.class_name_cn == "未知" and ml.inference_count < 30:
                    parts.append(f"P{pid}:校准{ml.inference_count}/30")
                else:
                    parts.append(f"P{pid}:{ml.class_name_cn}")
            if pred:
                parts.append(f"P{pid}_risk:{pred.risk_score:.0f}")
        summary = " | ".join(parts)
        cv2.putText(frame, summary, (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ════════════════════════════════════════════════
    # 工具方法 (共用)
    # ════════════════════════════════════════════════

    def _extract_landmarks(self, results) -> np.ndarray:
        lm = np.zeros((33, 4), dtype=np.float32)
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            lm[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
        return lm

    def _draw_skeleton(self, frame, results):
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2),
        )

    def _draw_low_confidence_skeleton(self, frame):
        if self.prev_lm is None:
            return
        h, w = frame.shape[:2]
        c = self.cfg.e2e_monitor.cyan
        lm = self.prev_lm
        for i in range(33):
            cx, cy = int(lm[i, 0] * w), int(lm[i, 1] * h)
            cv2.circle(frame, (cx, cy), 3, c, -1)
        for conn in self.mp_pose.POSE_CONNECTIONS:
            p1 = (int(lm[conn[0], 0] * w), int(lm[conn[0], 1] * h))
            p2 = (int(lm[conn[1], 0] * w), int(lm[conn[1], 1] * h))
            cv2.line(frame, p1, p2, c, 1)

    @staticmethod
    def _compute_torso_angle(lm: np.ndarray) -> float:
        shoulder_mid = (lm[11, :2] + lm[12, :2]) / 2
        hip_mid = (lm[23, :2] + lm[24, :2]) / 2
        dy = hip_mid[1] - shoulder_mid[1]
        dx = hip_mid[0] - shoulder_mid[0]
        return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))

    # ════════════════════════════════════════════════
    # 单人模式 HUD (v5.x 完整版, 保持不变)
    # ════════════════════════════════════════════════

    def _draw_hud_v4(self, frame, spatial, lm: np.ndarray,
                     ml_result=None, pred_report=None,
                     early_warn=None,
                     sedentary_sitting: bool = False,
                     med_msg: str = ""):
        """单人模式 HUD (v5.1 原版)"""
        h, w = frame.shape[:2]
        c = self.cfg.e2e_monitor
        font = cv2.FONT_HERSHEY_SIMPLEX

        panel_w = 320
        panel_h = 620
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (panel_w, panel_h), c.black, -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        y = 25

        # 当前动作
        if ml_result:
            action = ml_result.class_name_cn
            if ml_result.class_id == 0 and ml_result.fall_prob > 0.5:
                action_color = c.red
                action_bg = (0, 0, 60)
            elif ml_result.class_id == 0:
                action_color = c.orange
                action_bg = (0, 0, 30)
            elif ml_result.is_standing:
                action_color = (0, 255, 255)
                action_bg = (0, 40, 40)
            else:
                action_color = c.green
                action_bg = (0, 30, 0)

            cv2.rectangle(frame, (10, y), (panel_w - 10, y + 32), action_bg, -1)
            cv2.rectangle(frame, (10, y), (panel_w - 10, y + 32), action_color, 1)
            cv2.putText(frame, f"Current: {action}", (18, y + 24), font, 0.8, action_color, 2)
            y += 40

            # 六分类概率条
            probs = ml_result.probs
            class_labels_cn = ["摔倒", "坐下", "站起", "走路", "睡醒", "站立"]
            bar_colors = [
                (0, 0, 255), (255, 165, 0), (255, 255, 0),
                (0, 255, 0), (180, 130, 255), (0, 255, 255),
            ]
            cv2.putText(frame, "Activity Probs", (10, y), font, 0.45, c.white, 1)
            y += 18
            bar_start_x = 120
            bar_max_w = panel_w - bar_start_x - 20
            bar_h = 14
            bar_gap = 6
            for i in range(len(probs)):
                p = float(probs[i])
                cv2.putText(frame, f"{class_labels_cn[i]}", (10, y + bar_h - 2),
                            font, 0.38, bar_colors[i], 1)
                cv2.putText(frame, f"{p:.3f}", (bar_start_x - 55, y + bar_h - 2),
                            font, 0.32, c.gray, 1)
                cv2.rectangle(frame, (bar_start_x, y), (bar_start_x + bar_max_w, y + bar_h),
                              (40, 40, 40), -1)
                bar_fill = int(bar_max_w * p)
                if bar_fill > 0:
                    cv2.rectangle(frame, (bar_start_x, y),
                                  (bar_start_x + bar_fill, y + bar_h), bar_colors[i], -1)
                if i == ml_result.class_id:
                    cv2.rectangle(frame, (bar_start_x - 1, y - 1),
                                  (bar_start_x + bar_max_w + 1, y + bar_h + 1),
                                  bar_colors[i], 1)
                y += bar_h + bar_gap
            y += 6
            conf_color = c.red if ml_result.fall_confidence > 0.65 else (
                c.orange if ml_result.fall_confidence > 0.3 else c.green)
            cv2.putText(frame, f"Fall Conf: {ml_result.fall_confidence:.3f}",
                        (10, y), font, 0.42, conf_color, 1)
            y += 3
        y += 18

        # 空间运动
        cv2.putText(frame, "Motion Spatial", (10, y), font, 0.45, c.white, 1)
        y += 22
        _, risk = MotionSpatialAnalyzer.is_likely_fall(spatial, self.cfg.motion_spatial)
        risk_color = (c.red if risk >= 0.7 else c.orange if risk >= 0.5 else
                       c.yellow if risk >= 0.3 else c.green)
        cv2.putText(frame, f"  SpatialRisk: {risk:.2f}", (10, y), font, 0.45, risk_color, 1)
        y += 22

        # 久坐
        sed_status = self.sedentary.get_status()
        if sed_status["is_sitting"]:
            sed_min = sed_status["sitting_minutes"]
            sed_color = c.red if sed_min > 50 else (c.orange if sed_min > 40 else c.yellow)
            cv2.putText(frame, f"Sedentary: {sed_min:.0f}min", (10, y), font, 0.42, sed_color, 1)
        else:
            cv2.putText(frame, "Sedentary: Active", (10, y), font, 0.42, c.green, 1)
        y += 22

        # 跌倒预测
        if pred_report is not None:
            cv2.putText(frame, "Fall Prediction v2.0", (10, y), font, 0.45, c.white, 1)
            y += 18
            risk = pred_report.risk_score
            level_map = {"RED": (c.red, "!! RED !!"), "ORANGE": (c.orange, "ORANGE"),
                         "YELLOW": (c.yellow, "YELLOW")}
            risk_color, risk_label = level_map.get(pred_report.alert_level, (c.green, "SAFE"))
            cv2.putText(frame, f"  Risk: {risk:.0f}/100 [{risk_label}]",
                        (10, y), font, 0.42, risk_color, 1)
            y += 16
            if pred_report.suggestion:
                y += 16
                cv2.putText(frame, f"  > {pred_report.suggestion}",
                            (10, y), font, 0.28, c.yellow, 1)
        y += 20

        # 早期预警
        if early_warn is not None:
            cv2.putText(frame, "Fall Early Warning", (10, y), font, 0.45, c.white, 1)
            y += 18
            ew_risk = early_warn.pre_fall_risk
            if ew_risk >= 70:
                ew_risk_color, ew_risk_label = c.red, "!! CRITICAL !!"
            elif ew_risk >= 50:
                ew_risk_color, ew_risk_label = c.orange, "WARNING"
            elif ew_risk >= 30:
                ew_risk_color, ew_risk_label = c.yellow, "WATCH"
            else:
                ew_risk_color, ew_risk_label = c.green, "SAFE"
            cv2.putText(frame, f"  PreRisk: {ew_risk:.0f}/100 [{ew_risk_label}]",
                        (10, y), font, 0.42, ew_risk_color, 1)
            y += 16
            if early_warn.suggestion:
                cv2.putText(frame, f"  > {early_warn.suggestion}",
                            (10, y), font, 0.28, c.yellow, 1)

        # 帧计数
        cv2.putText(frame, f"frame: {self.frame_idx}", (w - 140, h - 10),
                    font, 0.4, c.gray, 1)

        return frame

    # ════════════════════════════════════════════════
    # 主循环 + 清理
    # ════════════════════════════════════════════════

    def run(self):
        mode_label = "多人检测" if self.use_multi_person else "单人检测"
        print(f"\n[4/5] 开始监控 ({mode_label})...")
        print(f"  {'─' * 55}")
        print(f"  按 Q 退出 | 六分类 + 跌倒早期预警 | {mode_label}")
        if self.use_multi_person:
            print(f"  策略: 任一人物触发跌倒风险 → 全局预警\n")
        else:
            print(f"  类别: 摔倒/坐下/站起/走路/睡醒/站立\n")

        window_name = self.cfg.e2e_monitor.window_name

        while True:
            ret, frame = self.cap.read()
            if not ret:
                if self._is_video_file:
                    print(f"\n  [INFO] 视频播放完毕")
                    break
                print("  [WARN] 丢帧，重试...")
                time.sleep(0.1)
                continue

            frame = self.process_frame(frame)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # cleanup() 由 main() 的 finally 统一调用，避免重复

    def cleanup(self):
        self.cap.release()
        if hasattr(self, 'pose'):
            try:
                self.pose.close()
            except (ValueError, AttributeError):
                pass
        if hasattr(self, 'person_tracker'):
            self.person_tracker.close()
        cv2.destroyAllWindows()
        self.logger.end_session()

        print(f"\n{'=' * 60}")
        print(f"  会话结束 | 总帧数: {self.frame_idx}")

        if self.use_multi_person:
            print(f"\n  [多人跟踪统计]")
            total_persons = len(self._person_states)
            active_persons = sum(1 for s in self._person_states.values() if s.total_frames > 10)
            yolo_ms = self.person_tracker.yolo_infer_ms if hasattr(self.person_tracker, 'yolo_infer_ms') else 0
            print(f"    YOLO 平均耗时: {yolo_ms:.0f}ms")
            print(f"    累计检测人物: {total_persons} 人 (其中 {active_persons} 人活跃 >10帧)")
            # 只显示活跃人物 (>10帧)
            active_states = {pid: s for pid, s in self._person_states.items() if s.total_frames > 10}
            for pid, state in sorted(active_states.items(), key=lambda x: x[1].total_frames, reverse=True)[:20]:
                if state.ml_detector:
                    stats = state.ml_detector.get_stats()
                    print(f"    P{pid:>3d}: {stats['stable_class_cn']:4s} "
                          f"({state.total_frames:>4d}帧)")
            # v6.1: 保存个性化基线 (只报告已校准的和失败的)
            saved = 0
            fail_count = 0
            calibrating = 0
            for pid, state in self._person_states.items():
                try:
                    if state.baseline.is_calibrated:
                        path = state.baseline.save()
                        saved += 1
                    else:
                        calibrating += 1
                except Exception as e:
                    fail_count += 1
                    print(f"    P{pid}: 基线保存失败: {e}")
            if saved > 0 or fail_count > 0:
                print(f"    个性化基线: {saved} 已保存, {fail_count} 失败, {calibrating} 未完成校准")
        else:
            report = self.gait_analyzer.force_analyze()
            if report:
                print(f"\n  [步态趋势]")
                print(f"    风险分: {self.gait_analyzer.risk_score:.1f}/100")
                print(f"    等级:   {self.gait_analyzer.alert_level}")
            if self.ml_detector:
                stats = self.ml_detector.get_stats()
                print(f"\n  [ML 六分类]")
                print(f"    推理次数: {stats['total_inferences']}")
                print(f"    平均耗时: {stats['avg_ms']}ms")
                print(f"    最终类别: {stats['stable_class_cn']}")
            # v6.1: 单人基线
            if hasattr(self, '_single_baseline') and self._single_baseline.is_calibrated:
                print(f"\n  [个性化步态基线]")
                print(f"    状态: 已校准 ({self._single_baseline._frame_count} 帧)")
                try:
                    self._single_baseline.save()
                    print(f"    基线已保存到 baselines/baseline_P0.json")
                except Exception as e:
                    print(f"    基线保存失败: {e}")

        print(f"{'=' * 60}")


# ════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="E2E Fall Monitor v6.3 — 六分类 + 多人 + 基线 + 踉跄前兆 + 行为画像")
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--model", "-m", type=str, default=_DEFAULT_MODEL)
    parser.add_argument("--no-ml", action="store_true",
                        help="禁用 ML 分类器")
    parser.add_argument("--source", "-s", type=str, default=None,
                        help="视频文件路径")
    parser.add_argument("--multi", "--multi-person", action="store_true",
                        dest="multi_person",
                        help="启用多人检测模式 (YOLOv8n + 逐人关键点)")
    parser.add_argument("--single", action="store_true",
                        help="强制单人模式 (默认)")
    parser.add_argument("--save-default", type=str, default=None)
    args = parser.parse_args()

    if args.save_default:
        cfg = FallConfig()
        cfg.save(args.save_default)
        print(f"默认配置已保存到: {args.save_default}")
        return

    if args.config:
        print(f"加载配置: {args.config}")
        cfg = FallConfig.load(args.config)
        print(f"  版本: {cfg.version} | {cfg.description}")
    else:
        cfg = FallConfig()

    # 确定模式
    use_multi = args.multi_person and not args.single

    print("=" * 60)
    mode_str = "多人检测" if use_multi else "单人检测"
    print(f"  E2E Fall Monitor v6.1 — 六分类 + 跌倒预测 + {mode_str} + 个性化基线")
    print("=" * 60)
    print(f"  ML 模型: {'禁用' if args.no_ml else args.model}")
    if use_multi:
        print(f"  多人模式: YOLOv8n + 逐人 MediaPipe + 独立管线")
        if not _MULTI_PERSON_AVAILABLE:
            print(f"  [ERROR] 多人检测模块不可用, 请先确保 src/multi_person.py 存在")
            return

    pipeline = FallMonitorPipeline(cfg, model_path=args.model,
                                    use_ml=not args.no_ml, source=args.source,
                                    use_multi_person=use_multi)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()
