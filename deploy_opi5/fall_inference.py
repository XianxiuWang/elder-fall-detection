#!/usr/bin/env python3
"""
fall_inference.py — 老人跌倒检测 视频推理管线
============================================
专为 Orange Pi 5 Pro (RK3588S ARM64) 优化
自包含版本：无外部模块依赖，EnhancedFeatureExtractor 内联

升级版: 51 维特征 (42 base + 9 temporal) + V5 确认锁 + 自由落体判别器
对应模型: fall_classifier_6class_v8.pkl (CV 90.44%, FPR 5.5/千窗 @ alpha=0.15/thresh=0.40)

用法:
  python3 fall_inference.py <video.mp4> [--save-video] [--model model.pkl]
  python3 fall_inference.py --realtime          # 实时摄像头模式
  python3 fall_inference.py --benchmark         # 性能基准测试

输出:
  results_<视频名>.json  每个窗口的详细推理结果
"""
import os, sys, time, json, pickle, argparse
import numpy as np
import cv2
from collections import deque, Counter
from typing import Tuple

# ============================================================
# 配置（根据 Orange Pi 5 Pro 性能可调）
# ============================================================
WINDOW_SIZE = 30       # 滑窗帧数
WINDOW_STRIDE = 6      # 滑窗步长
SKIP_FRAMES = 3        # 跳帧（每 N 帧取 1 帧做推理）
POSE_CHUNK = 200       # MediaPipe 分段重建间隔（防止内存泄漏）

# 时序过滤器参数（实时演示调优版）
VOTE_WINDOW = 10       # 多数投票窗口（7→10，更平滑）
FALL_HOLD = 15         # Fall 保持帧数
MIN_DURATION = 12      # 标签切换最少连续帧数（9→12，减少抖动）
FALL_PROB_THRESH = 0.40   # 阈值扫描最优 (alpha=0.15 下 recall 97.1% / FPR 5.5/千窗)
FALL_DESCENT_THRESH = 0.80   # person-heights/sec — 低于此值判为可控动作(弯腰/蹲下/坐下)

# Orange Pi 5 Pro 性能调优
MODEL_COMPLEXITY = 1   # MediaPipe Pose: 0=最快, 1=平衡, 2=最准
                       # RK3588S 建议用 1，实时可用 0

CLASS_NAMES = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"]
FALL_ID = 0
STANDING_ID = 5

COLORS_BGR = [
    (0, 0, 255), (0, 255, 128), (255, 128, 0),
    (255, 255, 0), (128, 0, 255), (128, 128, 128)
]

# ── 默认模型路径（板子上的相对路径） ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "models", "fall_classifier_6class_v8.pkl")


# ============================================================
# MediaPipe Pose 关键点索引
# ============================================================
NOSE = 0
LEFT_EYE_INNER = 1; RIGHT_EYE_INNER = 4
LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12
LEFT_ELBOW = 13; RIGHT_ELBOW = 14
LEFT_WRIST = 15; RIGHT_WRIST = 16
LEFT_HIP = 23; RIGHT_HIP = 24
LEFT_KNEE = 25; RIGHT_KNEE = 26
LEFT_ANKLE = 27; RIGHT_ANKLE = 28


# ============================================================
# 51 维特征提取器 (42 base + 9 temporal, 与 V8 训练一致)
# ============================================================
class EnhancedFeatureExtractor:
    """从关键点序列提取窗口级特征 — 51 维，自包含版本"""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size

    def extract_window(self, landmarks_seq: np.ndarray):
        """
        landmarks_seq: (window_size, 33, 3) 或 (window_size, 33, 4)
        返回: (51,) 特征向量
        """
        if len(landmarks_seq) < 3:
            return np.zeros(51, dtype=np.float32)

        lm = landmarks_seq.copy()
        T = len(lm)
        x = lm[:, :, 0]
        y = lm[:, :, 1]
        z = lm[:, :, 2] if lm.shape[2] >= 3 else np.zeros_like(x)
        vis = lm[:, :, 3] if lm.shape[2] >= 4 else np.ones_like(x)

        features = {}

        # ---- 1. 头部轨迹特征 (nose) ----
        nose_y = y[:, NOSE]
        nose_vy = np.gradient(nose_y)
        features["head_y_min"] = float(np.min(nose_y))
        features["head_y_max"] = float(np.max(nose_y))
        features["head_y_range"] = float(np.ptp(nose_y))
        features["head_y_drop"] = float(nose_y[0] - nose_y[-1])
        features["head_vy_max"] = float(np.max(np.abs(nose_vy)))
        features["head_vy_mean"] = float(np.mean(np.abs(nose_vy)))

        # ---- 2. 躯干特征 ----
        shoulder_mid_y = (y[:, LEFT_SHOULDER] + y[:, RIGHT_SHOULDER]) / 2
        hip_mid_y = (y[:, LEFT_HIP] + y[:, RIGHT_HIP]) / 2
        torso_center_y = (shoulder_mid_y + hip_mid_y) / 2
        features["torso_y_drop"] = float(torso_center_y[0] - torso_center_y[-1])
        features["torso_y_min"] = float(np.min(torso_center_y))
        features["torso_y_range"] = float(np.ptp(torso_center_y))

        shoulder_mid_x = (x[:, LEFT_SHOULDER] + x[:, RIGHT_SHOULDER]) / 2
        hip_mid_x = (x[:, LEFT_HIP] + x[:, RIGHT_HIP]) / 2
        dx = hip_mid_x - shoulder_mid_x
        dy = hip_mid_y - shoulder_mid_y + 1e-6
        torso_angles = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
        features["torso_angle_max"] = float(np.max(torso_angles))
        features["torso_angle_mean"] = float(np.mean(torso_angles))
        features["torso_angle_final"] = float(torso_angles[-1])
        features["torso_angle_change"] = float(np.ptp(torso_angles))

        # ---- 3. 运动重心位移 ----
        all_y_mid = np.mean(y[:, [LEFT_SHOULDER, RIGHT_SHOULDER,
                                   LEFT_HIP, RIGHT_HIP,
                                   LEFT_KNEE, RIGHT_KNEE]], axis=1)
        centroid_disp = np.sqrt(np.diff(all_y_mid, prepend=all_y_mid[0:1]) ** 2)
        features["centroid_disp_max"] = float(np.max(centroid_disp))
        features["centroid_disp_mean"] = float(np.mean(centroid_disp))
        features["centroid_disp_std"] = float(np.std(centroid_disp))
        features["centroid_total_disp"] = float(all_y_mid[-1] - all_y_mid[0])

        # ---- 4. 身体扩散范围 ----
        active_y_range = np.ptp(y, axis=1)
        features["spread_max"] = float(np.max(active_y_range))
        features["spread_mean"] = float(np.mean(active_y_range))
        features["spread_std"] = float(np.std(active_y_range))

        # ---- 5. 速度特征 ----
        centroid_vy = np.abs(np.gradient(all_y_mid))
        features["speed_max"] = float(np.max(centroid_vy))
        features["speed_mean"] = float(np.mean(centroid_vy))
        features["speed_std"] = float(np.std(centroid_vy))
        features["speed_peak_position"] = float(
            np.argmax(centroid_vy) / max(T, 1)
        ) if features["speed_max"] > 0 else 0.5
        accel = np.abs(np.gradient(centroid_vy))
        features["accel_max"] = float(np.max(accel))
        features["accel_mean"] = float(np.mean(accel))

        # ---- 6. 静止特征（首尾对比） ----
        tail_ratio = 0.25
        tail_n = max(1, int(T * tail_ratio))
        tail_frames = centroid_vy[-tail_n:]
        head_frames = centroid_vy[:tail_n]
        features["stillness_tail"] = float(np.mean(tail_frames))
        features["speed_ratio_tail_head"] = float(
            np.mean(tail_frames) / (np.mean(head_frames) + 1e-6)
        )

        # ---- 7. 下肢特征 ----
        knee_mid_y = (y[:, LEFT_KNEE] + y[:, RIGHT_KNEE]) / 2
        ankle_mid_y = (y[:, LEFT_ANKLE] + y[:, RIGHT_ANKLE]) / 2
        features["knee_y_drop"] = float(knee_mid_y[0] - knee_mid_y[-1])
        features["ankle_y_range"] = float(np.ptp(ankle_mid_y))
        features["hip_knee_ankle_spread_start"] = float(
            np.ptp([hip_mid_y[0], knee_mid_y[0], ankle_mid_y[0]])
        )
        features["hip_knee_ankle_spread_end"] = float(
            np.ptp([hip_mid_y[-1], knee_mid_y[-1], ankle_mid_y[-1]])
        )
        features["hip_knee_ankle_spread_min"] = float(
            min(np.min(hip_mid_y), np.min(knee_mid_y), np.min(ankle_mid_y))
        )

        # ---- 8. 上肢活动量 ----
        wrist_y_mean = (y[:, LEFT_WRIST] + y[:, RIGHT_WRIST]) / 2
        elbow_y_mean = (y[:, LEFT_ELBOW] + y[:, RIGHT_ELBOW]) / 2
        features["wrist_range"] = float(np.ptp(wrist_y_mean))
        features["elbow_wrist_dist"] = float(
            np.mean(np.abs(wrist_y_mean - elbow_y_mean))
        )

        # ---- 9. 头部-躯干偏移 ----
        features["head_torso_offset"] = float(
            np.mean(np.abs(nose_y - torso_center_y))
        )

        # ---- 10. 高级特征 ----
        centroid_vy_all = np.abs(np.gradient(all_y_mid))
        speed_weights = centroid_vy_all / (np.sum(centroid_vy_all) + 1e-6)
        features["motion_weighted_centroid"] = float(np.sum(speed_weights * all_y_mid))

        body_y_span = np.ptp(y, axis=1)
        start_n = min(5, T)
        end_n = min(5, T)
        features["active_range_start"] = float(np.mean(body_y_span[:start_n]))
        features["active_range_end"] = float(np.mean(body_y_span[-end_n:]))
        features["active_range_ratio"] = float(
            features["active_range_end"] / (features["active_range_start"] + 1e-6)
        )
        total_displacement = float(np.abs(all_y_mid[-1] - all_y_mid[0]))
        features["speed_distance_ratio"] = float(
            features["speed_max"] / (total_displacement + 1e-6)
        )
        features["body_compactness"] = float(
            np.mean(body_y_span) / (np.max(body_y_span) + 1e-6)
        )

        base_vec = np.array(list(features.values()), dtype=np.float32)

        # ---- 11. 时序增强特征（9 维）V3/V5 ----
        extra = []
        torso_y = y[:, 11]  # LEFT_SHOULDER y
        n_t = len(torso_y)
        extra.extend([
            torso_y[:n_t//3].mean(),
            torso_y[n_t//3:2*n_t//3].mean(),
            torso_y[2*n_t//3:].mean()
        ])
        x_arr = np.arange(n_t)
        extra.append(np.polyfit(x_arr, torso_y, 1)[0])
        extra.append(np.polyfit(x_arr, y[:, 0], 1)[0])  # nose y — y is 2D (T, 33) already
        speeds = np.linalg.norm(np.diff(lm[:, :, :2], axis=0), axis=2)
        extra.append(speeds.std(axis=0).mean())
        for idx in [0, 11, 23]:
            extra.append(lm[-1, idx, 1] - lm[0, idx, 1])

        # ---- 12. (V8: 已移除 8 维 tilt 特征, 与训练 extractor 对齐为 51 维) ----

        return np.concatenate([base_vec, np.array(extra, dtype=np.float32)])


# ============================================================
# 自由落体判别器 — 区分"真跌倒"和"弯腰/蹲下/坐下"
# ============================================================
def compute_max_descent_speed(window, effective_fps):
    """
    window: (30, 33, 3) keypoint array [frame, landmark, xyz]
    effective_fps: video_fps / SKIP_FRAMES (实际采样率)
    returns: max normalized trunk descent speed in person-heights/sec
             -1.0 = visible 太低, 不干预
    """
    # MediaPipe indices: 11=L.Shoulder, 12=R.Shoulder, 23=L.Hip, 24=R.Hip
    mid_hip_y = (window[:, 23, 1] + window[:, 24, 1]) / 2.0
    mid_shld_y = (window[:, 11, 1] + window[:, 12, 1]) / 2.0
    mid_shld_x = (window[:, 11, 0] + window[:, 12, 0]) / 2.0
    mid_hip_x = (window[:, 23, 0] + window[:, 24, 0]) / 2.0

    # 可见性检查 — 关键点不可靠则放弃判别
    hip_vis = (window[:, 23, 2] + window[:, 24, 2]) / 2.0
    if np.mean(hip_vis) < 0.3:
        return -1.0  # -1 = visible 太低, 不干预

    # 人体高度归一化 (前1/3帧躯干长度作为参考, 不受下降时拉伸影响)
    person_h = np.sqrt((mid_shld_x - mid_hip_x)**2 + (mid_shld_y - mid_hip_y)**2)
    ref_n = max(3, len(person_h) // 3)
    person_h_ref = float(np.median(person_h[:ref_n]))
    if person_h_ref < 0.015:  # 归一化坐标下, 躯干至少 ~1.5% 画面高
        return -1.0

    # 帧间下降速度 (正=向下)
    dt_per_step = 1.0 / effective_fps
    speeds = []
    for i in range(1, len(mid_hip_y)):
        dy = mid_hip_y[i] - mid_hip_y[i-1]
        if dy > 0:
            speeds.append(dy / person_h_ref / dt_per_step)

    if not speeds:
        return 0.0

    # 滑动窗口取最大 (捕获峰值坠落速度, 避免单帧噪声)
    win_frames = max(2, min(8, len(speeds)))
    max_speed = 0.0
    for i in range(len(speeds) - win_frames + 1):
        avg = np.mean(speeds[i:i+win_frames])
        if avg > max_speed:
            max_speed = avg

    return float(max_speed)


# ============================================================
# 时序过滤器 v3 + V5 确认锁 + 自由落体判别器
# ============================================================
class TemporalFilter:
    """时序过滤器 — V5 确认锁: 前 3 个窗口连续快速下降 → 锁定 Fall 事件"""

    def __init__(self, effective_fps=8.0):
        self.vote_win = VOTE_WINDOW
        self.fall_hold = FALL_HOLD
        self.min_dur = MIN_DURATION
        self.fall_thresh = FALL_PROB_THRESH
        self.effective_fps = effective_fps  # 用于计算下降速度
        self.pred_hist = deque(maxlen=VOTE_WINDOW * 2)
        self.fall_ctr = 0
        self.fall_recent = False
        self.fall_suppress = 0
        self.cur_label = -1
        self.label_dur = 0
        self.switch_pending = 0
        # V5 确认锁状态变量
        self.fall_confirmed = False      # 当前 Fall 事件是否已被快速下降锁定
        self.descent_ok_count = 0        # 连续快速下降窗口数 (≥3 锁定)
        # EMA 平滑 + 切换目标跟踪
        self._ema = None                 # EMA 缓存概率向量
        self._sw_target = -1             # 正在尝试切换到的目标标签

    def update(self, probs, window=None):
        # ── EMA 平滑原始概率（消除高频噪声，源头压下抖动） ──
        if self._ema is None:
            self._ema = probs.copy()
        alpha = 0.15  # EMA 平滑系数 (阈值扫描最优: 低 alpha → 误报↓, 略增延迟)
        self._ema = alpha * probs + (1 - alpha) * self._ema
        probs = self._ema.copy()
        
        raw = int(np.argmax(probs))
        fall_p = float(probs[FALL_ID])
        self.pred_hist.append(raw)
        vn = min(self.vote_win, len(self.pred_hist))
        recent = list(self.pred_hist)[-vn:]
        cnt = Counter(recent)
        top, _ = cnt.most_common(1)[0]

        # Fall 计数
        if top == FALL_ID:
            self.fall_ctr = min(self.fall_ctr + 1, self.fall_hold * 2)
            self.fall_recent = True
            self.fall_suppress = 30
        elif self.fall_ctr > 0:
            self.fall_ctr -= 1
            if self.fall_ctr == 0:
                self.fall_recent = False
        if self.fall_suppress > 0:
            self.fall_suppress -= 1

        # 抑制 WakeUp（Fall 刚结束时）
        if self.fall_suppress > 0 and top == 4:
            top = FALL_ID if fall_p > 0.2 else (
                self.cur_label if self.cur_label >= 0 else FALL_ID
            )

        # Fall hold + 投票校验（快速通道：不经过滞后，优先触发）
        fall_in_vote = cnt.get(FALL_ID, 0)
        if (self.fall_ctr > 0 and self.fall_ctr <= self.fall_hold
                and fall_p >= self.fall_thresh and fall_in_vote >= 2):
            final = FALL_ID
            self.cur_label = FALL_ID  # Fall 立即生效
        else:
            final = top
            
            # ── 滞后切换保护（70/30 阈值，仅非 Fall 场景） ──
            if self.cur_label >= 0 and top != self.cur_label:
                total = len(recent)
                new_ratio = cnt.get(top, 0) / total
                cur_ratio = cnt.get(self.cur_label, 0) / total
                if new_ratio > 0.7 and cur_ratio < 0.3:
                    self.cur_label = top
                else:
                    final = self.cur_label
            elif self.cur_label < 0:
                self.cur_label = final

        # ── V5 确认锁: 自由落体判别 ──
        # 新 Fall 事件必须连续 3 个窗口快速下降才能锁定, 锁定后不再检查
        # 解决: 真跌倒视频中低头/慢滑动窗口被误推翻 → 事件碎片化
        if self.cur_label == FALL_ID and window is not None:
            if not self.fall_confirmed:
                max_speed = compute_max_descent_speed(window, self.effective_fps)
                if max_speed >= 0 and max_speed >= FALL_DESCENT_THRESH:
                    self.descent_ok_count += 1
                    if self.descent_ok_count >= 3:
                        self.fall_confirmed = True  # 锁定: 后续事件内不再检查
                else:
                    # 下降太慢 → 减弱 Fall 置信度，但不强制翻转标签
                    # 让正常的投票+切换保护机制接管（避免标签闪烁）
                    self.descent_ok_count = 0
                    self.fall_ctr = max(0, self.fall_ctr - 4)
                    self.fall_recent = False
            # else: fall_confirmed → 跳过检查, 维持 Fall 连续性
        else:
            # 不在 Fall 状态 → 重置确认锁, 下次进入 Fall 需重新验证
            self.fall_confirmed = False
            self.descent_ok_count = 0

        return self.cur_label, float(probs[self.cur_label])

    def reset(self):
        self.pred_hist.clear()
        self.fall_ctr = 0
        self.fall_recent = False
        self.fall_suppress = 0
        self.cur_label = -1
        self.label_dur = 0
        self.switch_pending = 0
        self.fall_confirmed = False
        self.descent_ok_count = 0
        self._ema = None
        self._sw_target = -1


# ============================================================
# 主推理函数
# ============================================================
def load_model(model_path):
    """加载模型（兼容 V5 和 V6）"""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["scaler"], bundle


def run_inference(video_path, model_path=DEFAULT_MODEL, save_video=False,
                  show_fps=False):
    """视频文件推理 — 59维 V6 模型"""
    import mediapipe as mp  # 延迟导入，允许 --help 不依赖 mediapipe

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    # ── 加载模型 ──
    print("[1/5] Loading model...", flush=True)
    t0 = time.time()
    model, scaler, bundle = load_model(model_path)
    print(f"  Model: {bundle.get('config', {}).get('version', '?')}, "
          f"{bundle['feature_dim']} features, "
          f"accuracy={bundle.get('metrics', {}).get('accuracy', '?')}",
          flush=True)

    extractor = EnhancedFeatureExtractor(window_size=WINDOW_SIZE)

    # ── 打开视频 ──
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}", flush=True)
        sys.exit(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    effective_fps = fps / SKIP_FRAMES
    print(f"\n[2/5] Video: {os.path.basename(video_path)}", flush=True)
    print(f"  {total_frames} frames, {fps:.1f} fps, {width}x{height}", flush=True)
    print(f"  Duration: {total_frames/fps:.0f}s", flush=True)
    print(f"  Effective processing rate: {effective_fps:.1f} fps", flush=True)

    # ── 输出视频 ──
    out_writer = None
    if save_video:
        out_path = f"{base_name}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc,
                                     fps / SKIP_FRAMES, (width, height))

    # ── MediaPipe Pose + 推理 ──
    print(f"\n[3/5] Pose extraction + classification...", flush=True)
    print(f"  Model complexity: {MODEL_COMPLEXITY}", flush=True)
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4)

    keypoint_buffer = deque(maxlen=WINDOW_SIZE)
    filt = TemporalFilter(effective_fps=effective_fps)
    all_results = []

    frame_idx = 0
    processed = 0
    n_poses = 0
    frames_in_chunk = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SKIP_FRAMES != 0:
            frame_idx += 1
            continue

        # 分段重建 Pose（防止内存泄漏）
        if frames_in_chunk >= POSE_CHUNK:
            frames_in_chunk = 0
            pose.close()
            pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=MODEL_COMPLEXITY,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_mp = pose.process(rgb)

        if results_mp.pose_landmarks:
            kpts = np.array([[lm.x, lm.y, lm.visibility]
                             for lm in results_mp.pose_landmarks.landmark],
                            dtype=np.float32)
            keypoint_buffer.append(kpts)
            n_poses += 1

            if len(keypoint_buffer) == WINDOW_SIZE:
                window = np.array(keypoint_buffer, dtype=np.float32)
                vec = extractor.extract_window(window)
                vec_s = scaler.transform(vec.reshape(1, -1))
                probs = model.predict_proba(vec_s)[0]
                raw_label = int(np.argmax(probs))
                filt_label, confidence = filt.update(probs, window)

                all_results.append({
                    "frame": frame_idx,
                    "time_s": round(frame_idx / fps, 2),
                    "raw": CLASS_NAMES[raw_label],
                    "raw_conf": round(float(probs[raw_label]), 3),
                    "filt": CLASS_NAMES[filt_label],
                    "filt_conf": round(confidence, 3),
                    "fall_prob": round(float(probs[FALL_ID]), 3),
                })

                if save_video:
                    cv2.putText(frame, f"RAW: {CLASS_NAMES[raw_label]}",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, COLORS_BGR[raw_label], 2)
                    cv2.putText(frame, f"OUT: {CLASS_NAMES[filt_label]} ({confidence:.2f})",
                                (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, COLORS_BGR[filt_label], 3)
                    if filt_label == FALL_ID:
                        cv2.rectangle(frame, (0, 0), (width-1, height-1),
                                      (0, 0, 255), 6)
                    out_writer.write(frame)

                # 滑窗前进
                for _ in range(WINDOW_STRIDE):
                    if keypoint_buffer:
                        keypoint_buffer.popleft()

        processed += 1
        frames_in_chunk += 1
        frame_idx += 1

        if processed % 500 == 0:
            elapsed = time.time() - t0
            cur_fps = processed / elapsed if elapsed > 0 else 0
            eta = (total_frames / SKIP_FRAMES - processed) / cur_fps if cur_fps > 0 else 0
            print(f"  frame {frame_idx}/{total_frames} "
                  f"({frame_idx/total_frames*100:.0f}%), "
                  f"poses={n_poses}, preds={len(all_results)}, "
                  f"{cur_fps:.1f} f/s, ETA: {eta:.0f}s", flush=True)

    pose.close()
    cap.release()
    if out_writer:
        out_writer.release()

    elapsed = time.time() - t0
    n_preds = len(all_results)

    # ── 统计 ──
    print(f"\n[4/5] Processing complete: {n_preds} predictions in {elapsed:.1f}s",
          flush=True)
    print(f"  Average: {processed/elapsed:.1f} effective frames/s", flush=True)

    raw_counts = Counter(r["raw"] for r in all_results)
    filt_counts = Counter(r["filt"] for r in all_results)
    print(f"\n  Class distribution:", flush=True)
    print(f"  {'Class':<12s} {'Raw':>6s} {'Filtered':>9s}", flush=True)
    print(f"  {'-'*28}", flush=True)
    for name in CLASS_NAMES:
        print(f"  {name:<12s} {raw_counts.get(name,0):6d} "
              f"{filt_counts.get(name,0):9d}", flush=True)

    # Fall 事件检测
    fall_events = []
    for r in all_results:
        if r["filt"] == "Fall":
            t = r["time_s"]
            if not fall_events or t - fall_events[-1]["end"] > 2.0:
                fall_events.append({"start": t, "end": t})
            else:
                fall_events[-1]["end"] = t

    print(f"\n  Fall events: {len(fall_events)}", flush=True)
    for fe in fall_events:
        dur = fe["end"] - fe["start"]
        print(f"    {fe['start']:6.1f}s - {fe['end']:6.1f}s  "
              f"({dur:.1f}s)", flush=True)

    # ── 保存 JSON ──
    print(f"\n[5/5] Saving results...", flush=True)
    out_json = f"results_{base_name}.json"
    out_cfg = {
        "video": os.path.basename(video_path),
        "duration_s": total_frames / fps,
        "fps": fps,
        "platform": "Orange Pi 5 Pro",
        "model_version": bundle.get("config", {}).get("version", "V6"),
        "feature_dim": bundle["feature_dim"],
        "config": {
            "window_size": WINDOW_SIZE, "stride": WINDOW_STRIDE,
            "skip_frames": SKIP_FRAMES,
            "model_complexity": MODEL_COMPLEXITY,
            "vote_window": VOTE_WINDOW, "fall_hold": FALL_HOLD,
            "fall_prob_threshold": FALL_PROB_THRESH,
            "fall_descent_threshold": FALL_DESCENT_THRESH,
            "temporal_filter": "v3 + V5 confirmation lock + free-fall discriminator",
        },
        "stats": {
            "n_predictions": n_preds, "n_poses": n_poses,
            "raw_falls": raw_counts.get("Fall", 0),
            "filt_falls": filt_counts.get("Fall", 0),
            "fall_events": fall_events,
            "processing_fps": processed / elapsed,
        },
        "predictions": all_results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_cfg, f, ensure_ascii=False, indent=2)
    print(f"  JSON: {os.path.abspath(out_json)}", flush=True)
    if save_video:
        print(f"  Video: {os.path.abspath(out_path)}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  DONE — {elapsed:.0f}s total", flush=True)
    print(f"{'='*60}", flush=True)
    return all_results


def run_realtime(model_path=DEFAULT_MODEL, camera_id=0, camera_fps=30.0):
    """实时摄像头推理 — 59维 V6 模型"""
    import mediapipe as mp
    print("[实时模式] 加载模型...", flush=True)
    model, scaler, bundle = load_model(model_path)
    extractor = EnhancedFeatureExtractor(window_size=WINDOW_SIZE)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    effective_fps = camera_fps / SKIP_FRAMES
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4)

    keypoint_buffer = deque(maxlen=WINDOW_SIZE)
    filt = TemporalFilter(effective_fps=effective_fps)
    frame_idx = 0
    fall_alert = False

    print(f"[实时模式] 开始推理，effective_fps={effective_fps:.1f}, "
          f"按 'q' 退出...", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SKIP_FRAMES != 0:
            frame_idx += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_mp = pose.process(rgb)

        if results_mp.pose_landmarks:
            kpts = np.array([[lm.x, lm.y, lm.visibility]
                             for lm in results_mp.pose_landmarks.landmark],
                            dtype=np.float32)
            keypoint_buffer.append(kpts)

            if len(keypoint_buffer) == WINDOW_SIZE:
                window = np.array(keypoint_buffer, dtype=np.float32)
                vec = extractor.extract_window(window)
                vec_s = scaler.transform(vec.reshape(1, -1))
                probs = model.predict_proba(vec_s)[0]
                filt_label, conf = filt.update(probs, window)

                # UI 显示
                label_text = f"{CLASS_NAMES[filt_label]} ({conf:.2f})"
                color = COLORS_BGR[filt_label]
                cv2.putText(frame, label_text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                if filt_label == FALL_ID:
                    if not fall_alert:
                        print(f"\n!! FALL DETECTED at frame {frame_idx}!", flush=True)
                        fall_alert = True
                    cv2.rectangle(frame, (0, 0),
                                  (frame.shape[1]-1, frame.shape[0]-1),
                                  (0, 0, 255), 6)
                    cv2.putText(frame, "!! FALL!", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                else:
                    fall_alert = False

                for _ in range(WINDOW_STRIDE):
                    if keypoint_buffer:
                        keypoint_buffer.popleft()

        cv2.imshow("Fall Detection (Orange Pi 5 Pro)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    pose.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[实时模式] 已退出", flush=True)


def run_benchmark(model_path=DEFAULT_MODEL, num_frames=300):
    """性能基准测试（无视频 I/O，纯推理速度）"""
    import mediapipe as mp

    print(f"[基准测试] 加载模型...", flush=True)
    model, scaler, bundle = load_model(model_path)
    extractor = EnhancedFeatureExtractor(window_size=WINDOW_SIZE)

    # 生成模拟关键点（640x480 空间）
    print(f"[基准测试] 生成 {num_frames} 帧模拟数据...", flush=True)
    rng = np.random.RandomState(42)
    dummy_frames = []
    for i in range(num_frames):
        # 模拟站姿关键点
        kpts = np.zeros((33, 3), dtype=np.float32)
        kpts[:, 0] = rng.normal(0.5, 0.1, 33).clip(0, 1)
        kpts[:, 1] = rng.normal(0.5, 0.15, 33).clip(0, 1)
        kpts[:, 2] = rng.uniform(0.5, 1.0, 33)
        dummy_frames.append(kpts)

    print(f"[基准测试] 预热 MediaPipe...", flush=True)
    mp.solutions.pose.Pose(static_image_mode=False, model_complexity=MODEL_COMPLEXITY)

    print(f"[基准测试] 运行推理 {num_frames} 帧...", flush=True)
    keypoint_buffer = deque(maxlen=WINDOW_SIZE)
    filt = TemporalFilter(effective_fps=30.0 / SKIP_FRAMES)

    t0 = time.time()
    n_preds = 0
    for kpts in dummy_frames:
        keypoint_buffer.append(kpts)
        if len(keypoint_buffer) == WINDOW_SIZE:
            window = np.array(keypoint_buffer, dtype=np.float32)
            vec = extractor.extract_window(window)
            vec_s = scaler.transform(vec.reshape(1, -1))
            probs = model.predict_proba(vec_s)[0]
            filt.update(probs, window)
            n_preds += 1
            for _ in range(WINDOW_STRIDE):
                if keypoint_buffer:
                    keypoint_buffer.popleft()

    elapsed = time.time() - t0

    print(f"\n{'='*50}", flush=True)
    print(f"  基准测试结果 (Orange Pi 5 Pro)", flush=True)
    print(f"  {'─'*40}", flush=True)
    print(f"  特征维度:       {bundle['feature_dim']}", flush=True)
    print(f"  模型复杂度:     {MODEL_COMPLEXITY}", flush=True)
    print(f"  滑窗大小:       {WINDOW_SIZE}", flush=True)
    print(f"  跳帧:           每 {SKIP_FRAMES} 帧取 1", flush=True)
    print(f"  总帧数:         {num_frames}", flush=True)
    print(f"  推理次数:       {n_preds}", flush=True)
    print(f"  总耗时:         {elapsed:.2f}s", flush=True)
    print(f"  推理速度:       {n_preds/elapsed:.1f} preds/s", flush=True)
    print(f"  有效帧率:       {num_frames/elapsed:.1f} fps", flush=True)
    print(f"  每推理耗时:     {elapsed/n_preds*1000:.1f} ms", flush=True)
    print(f"{'='*50}", flush=True)


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="老人跌倒检测 — Orange Pi 5 Pro 部署版 (59维 V6 + V5确认锁)")
    parser.add_argument("video", nargs="?", help="输入视频路径")
    parser.add_argument("--save-video", action="store_true",
                        help="输出标注视频")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"模型路径 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--complexity", type=int, default=MODEL_COMPLEXITY,
                        choices=[0, 1, 2],
                        help="MediaPipe 模型复杂度 (0=最快, 1=平衡, 2=最准)")
    parser.add_argument("--realtime", action="store_true",
                        help="实时摄像头模式")
    parser.add_argument("--camera", type=int, default=0,
                        help="摄像头 ID (默认 0)")
    parser.add_argument("--camera-fps", type=float, default=30.0,
                        help="摄像头帧率 (默认 30)")
    parser.add_argument("--benchmark", action="store_true",
                        help="性能基准测试")
    parser.add_argument("--bench-frames", type=int, default=300,
                        help="基准测试帧数 (默认 300)")

    args = parser.parse_args()

    if args.complexity != MODEL_COMPLEXITY:
        MODEL_COMPLEXITY = args.complexity

    if args.benchmark:
        run_benchmark(model_path=args.model,
                      num_frames=args.bench_frames)
    elif args.realtime:
        run_realtime(model_path=args.model, camera_id=args.camera,
                     camera_fps=args.camera_fps)
    elif args.video:
        run_inference(args.video, model_path=args.model,
                      save_video=args.save_video)
    else:
        parser.print_help()
