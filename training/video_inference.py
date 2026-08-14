#!/usr/bin/env python3
"""
video_inference.py — 真实视频流推理管线（带时序过滤器）
========================================================
完整流程:
  RGB 视频 → MediaPipe Pose → 30帧滑窗 → V5 分类器 → 时序过滤器 → JSON/视频

用法:
  python video_inference.py "F:\动作数据集\50 Ways to Fall.mp4"
  python video_inference.py "F:\动作数据集\50 Ways to Stand.mp4" --save-video

输出:
  results_<视频名>.json   每窗口 (时间, 原始标签, 过滤标签, 置信度, fall概率)
"""
import os, sys, time, json, pickle
import numpy as np
import cv2
import mediapipe as mp
from collections import deque, Counter

# 导入训练代码的特征提取器
sys.path.insert(0, r"E:\老人跌倒\training")
from train_fall_classifier import FeatureExtractor

MODEL_PATH = r"E:\老人跌倒\models\fall_classifier_6class_v6.pkl"
CLASS_NAMES = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"]
FALL_ID = 0
STANDING_ID = 5

WINDOW_SIZE = 30
WINDOW_STRIDE = 6
SKIP_FRAMES = 3
POSE_CHUNK = 200

# 时序过滤器参数
VOTE_WINDOW = 7
FALL_HOLD = 15
MIN_DURATION = 9
FALL_PROB_THRESH = 0.35
FALL_DESCENT_THRESH = 0.80     # person-heights/sec — 低于此值判为可控动作(弯腰/蹲下/坐下)

COLORS_BGR = [
    (0, 0, 255), (0, 255, 128), (255, 128, 0),
    (255, 255, 0), (128, 0, 255), (128, 128, 128)
]


# ============================================================
# 59 维特征提取器 (42 base + 9 temporal + 8 tilt, 与 V6 训练一致)
# ============================================================
class EnhancedFeatureExtractor(FeatureExtractor):
    def __init__(self, window_size=WINDOW_SIZE):
        super().__init__(window_size=window_size)
    
    def extract_window(self, window):
        base_vec, base_names = super().extract_window(window)
        extra = []
        n = len(window)
        
        # ── 9 temporal features (V3/V5) ──
        torso_y = window[:, 11, 1]
        n_t = len(torso_y)
        extra.extend([torso_y[:n_t//3].mean(), torso_y[n_t//3:2*n_t//3].mean(), torso_y[2*n_t//3:].mean()])
        x = np.arange(n_t)
        extra.append(np.polyfit(x, torso_y, 1)[0])
        extra.append(np.polyfit(x, window[:, 0, 1], 1)[0])
        speeds = np.linalg.norm(np.diff(window[:, :, :2], axis=0), axis=2)
        extra.append(speeds.std(axis=0).mean())
        for idx in [0, 11, 23]:
            extra.append(window[-1, idx, 1] - window[0, idx, 1])
        
        # ── 8 tilt/body-orientation features (V6) ──
        shoulder_mid_y = (window[:, 11, 1] + window[:, 12, 1]) / 2
        hip_mid_y = (window[:, 23, 1] + window[:, 24, 1]) / 2
        shoulder_mid_x = (window[:, 11, 0] + window[:, 12, 0]) / 2
        hip_mid_x = (window[:, 23, 0] + window[:, 24, 0]) / 2
        head_y = window[:, 0, 1]
        head_x = window[:, 0, 0]
        
        late_n = min(5, n)
        # 1: torso_tilt_late
        dx_late = hip_mid_x[-late_n:] - shoulder_mid_x[-late_n:]
        dy_late = hip_mid_y[-late_n:] - shoulder_mid_y[-late_n:] + 1e-6
        tilt_late = np.degrees(np.arctan2(np.abs(dx_late), np.abs(dy_late)))
        extra.append(float(np.mean(tilt_late)))
        # 2: torso_tilt_delta
        early_n = min(5, n)
        dx_early = hip_mid_x[:early_n] - shoulder_mid_x[:early_n]
        dy_early = hip_mid_y[:early_n] - shoulder_mid_y[:early_n] + 1e-6
        tilt_early = np.degrees(np.arctan2(np.abs(dx_early), np.abs(dy_early)))
        extra.append(float(np.mean(tilt_late) - np.mean(tilt_early)))
        # 3: head_below_hip_ratio
        extra.append(float(np.mean(head_y[-late_n:] > hip_mid_y[-late_n:])))
        # 4: torso_rel_height
        torso_center_y = (shoulder_mid_y + hip_mid_y) / 2
        body_y_max = np.max(window[:, :, 1])
        body_y_min = np.min(window[:, :, 1])
        body_y_span = body_y_max - body_y_min + 1e-6
        extra.append(float(np.clip(np.mean((body_y_max - torso_center_y[-late_n:]) / body_y_span), 0, 1)))
        # 5: horizontal_spread_ratio
        body_x_span = np.max(window[:, :, 0]) - np.min(window[:, :, 0])
        extra.append(float(np.clip(body_x_span / (body_y_span + 1e-6), 0, 10)))
        # 6: torso_angle_accel
        all_tilt = np.degrees(np.arctan2(
            np.abs(hip_mid_x - shoulder_mid_x),
            np.abs(hip_mid_y - shoulder_mid_y) + 1e-6))
        tilt_vel = np.gradient(all_tilt)
        tilt_accel = np.gradient(tilt_vel)
        extra.append(float(np.max(np.abs(tilt_accel))))
        # 7: shoulder_hip_shift
        extra.append(float(np.mean(np.abs(shoulder_mid_x - hip_mid_x)[-late_n:])))
        # 8: kp_above_hips_ratio
        extra.append(float(np.mean(window[-late_n:, :, 1] < hip_mid_y[-late_n:, np.newaxis])))
        
        return np.concatenate([base_vec, np.array(extra)])


# ============================================================
# 自由落体判别器 — 区分"真跌倒"和"弯腰/蹲下/坐下"
# ============================================================
# 原理: 跌倒时躯干是被动坠落(接近重力加速度), 弯腰/蹲下是主动控制(远低于重力加速度)
# 将像素速度归一化为 person-heights/sec, 阈值 0.50 源自:
#   - 站立→倒地(0.6s自由落体): ~1.5 heights/sec
#   - 弯腰捡东西(1.0s): ~0.4 heights/sec
#   - 深蹲(1.5s): ~0.3 heights/sec
#   - 滑落椅子(1.2s): ~0.6 heights/sec  ← 边界, 用 0.50 可捕获

def compute_max_descent_speed(window, effective_fps):
    """
    window: (30, 33, 3) keypoint array [frame, landmark, xyz]
    effective_fps: video_fps / SKIP_FRAMES (实际采样率)
    returns: max normalized trunk descent speed in person-heights/sec
    """
    import numpy as np

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

    # 5帧滑动窗口取最大 (捕获峰值坠落速度, 避免单帧噪声)
    win_frames = max(2, min(8, len(speeds)))
    max_speed = 0.0
    for i in range(len(speeds) - win_frames + 1):
        avg = np.mean(speeds[i:i+win_frames])
        if avg > max_speed:
            max_speed = avg

    return float(max_speed)


# ============================================================
# 时序过滤器
# ============================================================
class TemporalFilter:
    """时序过滤器 — 关键改进：Fall 优先，防止 WakeUp 淹没 Fall + 自由落体判别"""
    def __init__(self, effective_fps=8.0):
        self.vote_win = VOTE_WINDOW
        self.fall_hold = FALL_HOLD
        self.min_dur = MIN_DURATION
        self.fall_thresh = FALL_PROB_THRESH
        self.effective_fps = effective_fps  # 用于计算下降速度
        self.pred_hist = deque(maxlen=VOTE_WINDOW * 2)
        self.fall_ctr = 0
        self.fall_recent = False     # 最近是否有 Fall（用于抑制 WakeUp）
        self.fall_suppress = 0       # 抑制 WakeUp 的剩余帧数
        self.cur_label = -1
        self.label_dur = 0
        self.switch_pending = 0
        self.fall_confirmed = False    # 当前 Fall 事件是否已被快速下降锁定
        self.descent_ok_count = 0      # 连续快速下降窗口数 (≥3 锁定)

    def update(self, probs, window=None):
        raw = int(np.argmax(probs))
        fall_p = float(probs[FALL_ID])
        self.pred_hist.append(raw)
        vn = min(self.vote_win, len(self.pred_hist))
        recent = list(self.pred_hist)[-vn:]
        cnt = Counter(recent)
        top, _ = cnt.most_common(1)[0]
        
        # Fall 检测：Fall 计数
        if top == FALL_ID:
            self.fall_ctr = min(self.fall_ctr + 1, self.fall_hold * 2)
            self.fall_recent = True
            self.fall_suppress = 30  # 抑制 WakeUp 30 帧 (~5秒)
        elif self.fall_ctr > 0:
            self.fall_ctr -= 1
            if self.fall_ctr == 0:
                self.fall_recent = False
        if self.fall_suppress > 0:
            self.fall_suppress -= 1
        
        # 如果最近有 Fall，强制抑制 WakeUp
        if self.fall_suppress > 0 and top == 4:  # 4 = WakeUp
            # 改为 Fall（如果 fall_p 足够高）或保持上次标签
            if fall_p > 0.2:
                top = FALL_ID
            else:
                top = self.cur_label if self.cur_label >= 0 else FALL_ID
        
        # Fall hold：检测到 Fall 后保持（需要当前窗口也有 Fall 支撑）
        fall_in_vote = cnt.get(FALL_ID, 0)
        if self.fall_ctr > 0 and self.fall_ctr <= self.fall_hold and fall_p >= self.fall_thresh and fall_in_vote >= 2:
            final = FALL_ID
        else:
            final = top
        
        # 类别切换保护（修复：累计"想切换"的次数，不是原地死循环）
        if final != self.cur_label and self.cur_label >= 0:
            self.switch_pending += 1
            if self.switch_pending < self.min_dur:
                final = self.cur_label  # 保持旧标签
            else:
                # 切换批准
                self.switch_pending = 0
                self.label_dur = 0
                self.cur_label = final
        else:
            if self.cur_label != final:
                self.label_dur = 0
                self.cur_label = final if self.cur_label < 0 else self.cur_label
            else:
                self.label_dur += 1
            self.switch_pending = 0
        
        # ── 自由落体判别 (V5): 确认锁机制 ──
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
                    # 下降太慢 → 不是真跌倒, 推翻并重置计数
                    self.descent_ok_count = 0
                    recent_full = list(self.pred_hist)
                    vn_full = min(self.vote_win, len(recent_full))
                    non_fall = [l for l in recent_full[-vn_full:] if l != FALL_ID]
                    if non_fall:
                        alt = Counter(non_fall).most_common(1)[0][0]
                    else:
                        alt = STANDING_ID
                    self.cur_label = alt
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


# ============================================================
# 主推理管线
# ============================================================
def run_inference(video_path, save_video=False):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # ── 加载模型 ──
    print("[1/5] Loading model...")
    t0 = time.time()
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    scaler = bundle["scaler"]
    print(f"  Model: {bundle.get('config',{}).get('version','?')}, "
          f"{bundle['feature_dim']} features, accuracy={bundle.get('metrics',{}).get('accuracy','?')}")
    
    extractor = EnhancedFeatureExtractor(window_size=WINDOW_SIZE)
    
    # ── 打开视频 ──
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n[2/5] Video: {os.path.basename(video_path)}")
    print(f"  {total_frames} frames, {fps:.1f} fps, {width}x{height}")
    print(f"  Duration: {total_frames/fps:.0f}s")
    
    out_writer = None
    if save_video:
        out_path = f"{base_name}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps / SKIP_FRAMES, (width, height))
    
    # ── MediaPipe Pose + 推理 ──
    print(f"\n[3/5] Pose extraction + classification...")
    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.4, min_tracking_confidence=0.4)
    
    keypoint_buffer = deque(maxlen=WINDOW_SIZE)
    filt = TemporalFilter(effective_fps=fps / SKIP_FRAMES)
    all_results = []
    
    frame_idx = 0
    processed = 0
    frames_in_chunk = 0
    n_poses = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % SKIP_FRAMES != 0:
            frame_idx += 1
            continue
        
        # 分段重建 Pose
        if frames_in_chunk >= POSE_CHUNK:
            frames_in_chunk = 0
            pose.close()
            pose = mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=1,
                min_detection_confidence=0.4, min_tracking_confidence=0.4)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_mp = pose.process(rgb)
        
        if results_mp.pose_landmarks:
            kpts = np.array([[lm.x, lm.y, lm.visibility]
                             for lm in results_mp.pose_landmarks.landmark], dtype=np.float32)
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
                    cv2.putText(frame, f"RAW: {CLASS_NAMES[raw_label]}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS_BGR[raw_label], 2)
                    cv2.putText(frame, f"OUT: {CLASS_NAMES[filt_label]} ({confidence:.2f})", (10, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS_BGR[filt_label], 3)
                    if filt_label == FALL_ID:
                        cv2.rectangle(frame, (0, 0), (width-1, height-1), (0, 0, 255), 6)
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
            print(f"  frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.0f}%), "
                  f"poses={n_poses}, preds={len(all_results)}, {processed/elapsed:.0f} f/s")
    
    pose.close()
    cap.release()
    if out_writer:
        out_writer.release()
    
    elapsed = time.time() - t0
    n_preds = len(all_results)
    
    # ── 统计 ──
    print(f"\n[4/5] Processing complete: {n_preds} predictions in {elapsed:.1f}s")
    
    raw_counts = Counter(r["raw"] for r in all_results)
    filt_counts = Counter(r["filt"] for r in all_results)
    
    print(f"\n  Class distribution:")
    print(f"  {'Class':<12s} {'Raw':>6s} {'Filtered':>9s}")
    print(f"  {'-'*28}")
    for name in CLASS_NAMES:
        print(f"  {name:<12s} {raw_counts.get(name,0):6d} {filt_counts.get(name,0):9d}")
    
    # 识别 Fall 事件（连续 Fall 帧合并为一个事件）
    fall_events = []
    for r in all_results:
        if r["filt"] == "Fall":
            t = r["time_s"]
            if not fall_events or t - fall_events[-1]["end"] > 2.0:
                fall_events.append({"start": t, "end": t})
            else:
                fall_events[-1]["end"] = t
    
    print(f"\n  Fall events: {len(fall_events)}")
    for fe in fall_events:
        dur = fe["end"] - fe["start"]
        label = "REAL?" if dur < 10 else "LONG (suspicious)"
        print(f"    {fe['start']:6.1f}s - {fe['end']:6.1f}s  ({dur:.1f}s)  {label}")
    
    # ── 保存 JSON ──
    print(f"\n[5/5] Saving results...")
    out_json = f"results_{base_name}.json"
    out_cfg = {
        "video": os.path.basename(video_path),
        "duration_s": total_frames / fps,
        "fps": fps,
        "config": {
            "window_size": WINDOW_SIZE, "stride": WINDOW_STRIDE,
            "skip_frames": SKIP_FRAMES,
            "vote_window": VOTE_WINDOW, "fall_hold": FALL_HOLD,
            "fall_prob_threshold": FALL_PROB_THRESH,
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
    print(f"  JSON: {os.path.abspath(out_json)}")
    if save_video:
        print(f"  Video: {os.path.abspath(out_path)}")
    
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    return all_results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python video_inference.py <video.mp4> [--save-video]")
        sys.exit(1)
    
    video = sys.argv[1]
    save = "--save-video" in sys.argv
    
    if not os.path.exists(video):
        print(f"ERROR: {video} not found")
        sys.exit(1)
    
    run_inference(video, save)
