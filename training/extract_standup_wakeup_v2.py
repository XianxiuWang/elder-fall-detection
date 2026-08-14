#!/usr/bin/env python3
"""
extract_standup_wakeup_v2.py — 分段提取 StandUp/WakeUp 样本（v2）
================================================================
改进点（相比 v1 extract_standup_wakeup.py）：
  1. SKIP_FRAMES=1 不跳帧，数据量约为 v1 的 6 倍
  2. 真正的动作分段：
     - 姿态丢失 gap（转场/人物出画）→ 硬边界切段
     - 段内长静帧（动作间停顿复位）→ 软边界切段
  3. 每段独立存 .npz，命名 standup_XXXX.npz / wakeup_XXXX.npz

输出: E:\老人跌倒\data\custom_standup_wakeup\
  keypoints=(n_frames, 33, 4)  [x, y, z, visibility]  ← 与 custom_6class 主格式一致
  category=2(StandUp) / 4(WakeUp)
"""
import os, sys, time
import numpy as np
import cv2
import mediapipe as mp

# ─── 配置 ───
VIDEOS = [
    (r"F:\动作数据集\50 Ways to Stand -50种站立方式.mp4", 2, "standup"),
    (r"F:\动作数据集\50 Ways to Wake Up50种睡醒方式.mp4", 4, "wakeup"),
]
OUTPUT_DIR = r"E:\老人跌倒\data\custom_standup_wakeup"
LOG_FILE = r"E:\老人跌倒\data\extraction_standup_wakeup_v2_log.txt"

SKIP_FRAMES = 1          # 不跳帧（v1=6，砍了5/6数据）
POSE_CHUNK = 300         # 每 N 帧重建 Pose，防状态退化
MIN_DETECTION_CONF = 0.4
MIN_TRACKING_CONF = 0.4

# ─── 分段参数 ───
GAP_THRESHOLD = 4        # 连续丢失姿态 >= 4 帧 → 切段（硬边界）
STILL_MOVEMENT = 0.005   # 平均关节位移(归一化) < 此值 = 静止
STILL_WINDOW = 12        # 连续静止 >= 12 帧 → 切段（软边界）
MIN_SEG_FRAMES = 30      # 最短保留段（>= 训练窗口 30）

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


def create_pose():
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_TRACKING_CONF,
    )


# ================================================================
# 1) 逐帧提取关键点（返回 [(frame_idx, kpts_33x4 或 None), ...]）
# ================================================================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    log(f"\n{'='*60}")
    log(f"  Video: {os.path.basename(video_path)}")
    log(f"  Frames: {total_frames}, FPS: {fps:.1f}")

    frames = []            # (frame_idx, kpts or None)
    frame_idx = 0
    frames_in_chunk = 0
    chunk_idx = 0
    t_start = time.time()
    pose = create_pose()
    n_valid = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SKIP_FRAMES != 0:
            frame_idx += 1
            continue

        if frames_in_chunk >= POSE_CHUNK:
            frames_in_chunk = 0
            chunk_idx += 1
            pose.close()
            pose = create_pose()
            log(f"  ── chunk {chunk_idx} reset @ frame {frame_idx} ──")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            kpts = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark],
                dtype=np.float32)
            frames.append((frame_idx, kpts))
            n_valid += 1
        else:
            frames.append((frame_idx, None))

        frames_in_chunk += 1
        frame_idx += 1

        if frame_idx % 800 == 0:
            elapsed = time.time() - t_start
            pct = frame_idx / total_frames * 100
            log(f"    frame {frame_idx}/{total_frames} ({pct:.1f}%), "
                f"valid={n_valid}, {frame_idx/elapsed:.0f} f/s")

    pose.close()
    cap.release()
    elapsed = time.time() - t_start
    log(f"  Done: {len(frames)} frames, {n_valid} valid poses ({elapsed:.1f}s)")
    return frames, fps


# ================================================================
# 2) 分段：姿态丢失 gap（硬边界）→ 段内静帧（软边界）
# ================================================================
def movement_between(k1, k2):
    """两帧间可见关节的平均位移（归一化坐标）。"""
    valid = (k1[:, 3] > 0.5) & (k2[:, 3] > 0.5)
    if valid.sum() < 10:
        return 0.0
    return float(np.linalg.norm(k2[valid, :2] - k1[valid, :2], axis=1).mean())


def segment_frames(frames):
    """frames: [(frame_idx, kpts or None)] → list[list[(frame_idx, kpts)]]"""
    # 第一步：按姿态丢失 gap 切出"有效段"
    raw_segments = []
    cur = []
    gap = 0
    for fi, kpts in frames:
        if kpts is not None:
            if cur and gap >= GAP_THRESHOLD:
                raw_segments.append(cur)
                cur = []
            cur.append((fi, kpts))
            gap = 0
        else:
            gap += 1
    if cur:
        raw_segments.append(cur)

    # 第二步：在长段内按静帧（动作间停顿）再切
    final_segments = []
    for seg in raw_segments:
        if len(seg) < STILL_WINDOW + 2:
            final_segments.append(seg)
            continue
        # 计算段内相邻帧位移
        moves = [0.0]
        for i in range(1, len(seg)):
            moves.append(movement_between(seg[i-1][1], seg[i][1]))
        moves = np.array(moves)

        # 找连续静止窗口（位移 < STILL_MOVEMENT）
        still = moves < STILL_MOVEMENT
        # 找 >= STILL_WINDOW 的静止段，在其中点切开
        split_points = []
        i = 0
        while i < len(still):
            if still[i]:
                j = i
                while j < len(still) and still[j]:
                    j += 1
                if j - i >= STILL_WINDOW:
                    split_points.append((i + j) // 2)
                i = j
            else:
                i += 1

        # 按切点切段
        if not split_points:
            final_segments.append(seg)
        else:
            prev = 0
            for sp in split_points:
                if sp - prev >= MIN_SEG_FRAMES:
                    final_segments.append(seg[prev:sp])
                prev = sp
            if len(seg) - prev >= MIN_SEG_FRAMES:
                final_segments.append(seg[prev:])

    # 过滤过短段
    final_segments = [s for s in final_segments if len(s) >= MIN_SEG_FRAMES]
    return final_segments


# ================================================================
# 3) 保存
# ================================================================
def save_segments(segments, class_id, prefix):
    lens = sorted([len(s) for s in segments])
    log(f"  → 切出 {len(segments)} 段，帧数分布: min={lens[0] if lens else 0}, "
        f"median={np.median(lens) if lens else 0:.0f}, max={lens[-1] if lens else 0}")

    saved = 0
    for seg in segments:
        arr = np.array([k for _, k in seg], dtype=np.float32)  # (n, 33, 4)
        out = os.path.join(OUTPUT_DIR, f"{prefix}_{saved:04d}.npz")
        np.savez_compressed(
            out,
            keypoints=arr,
            category=class_id,
            n_frames=len(seg),
            source_video=os.path.basename(VIDEOS[0][0] if class_id == 2 else VIDEOS[1][0]),
        )
        saved += 1
    return saved


if __name__ == "__main__":
    log(f"{'='*60}")
    log(f"  StandUp + WakeUp 分段提取 (v2)")
    log(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  SKIP={SKIP_FRAMES}, GAP={GAP_THRESHOLD}, STILL={STILL_MOVEMENT}/{STILL_WINDOW}")
    log(f"{'='*60}")

    total_files = 0
    for video_path, class_id, prefix in VIDEOS:
        if not os.path.exists(video_path):
            log(f"  ⚠ NOT FOUND: {video_path}")
            continue
        frames, fps = process_video(video_path)
        segments = segment_frames(frames)
        n = save_segments(segments, class_id, prefix)
        total_files += n

    log(f"\n{'='*60}")
    log(f"  COMPLETE: {total_files} 个独立动作段保存到 {OUTPUT_DIR}")
    log(f"{'='*60}")
