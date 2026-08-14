#!/usr/bin/env python3
"""
extract_sitdown.py — 分段提取 SitDown 样本（50个坐下的动作.mp4）
====================================================================
照搬 extract_standup_wakeup_v2.py 的分段逻辑：
  1. SKIP_FRAMES=1 不跳帧
  2. 姿态丢失 gap（转场/人物出画）→ 硬边界切段
  3. 段内长静帧（坐下后保持坐姿的停顿）→ 软边界切段
  4. 每段独立存 .npz，命名 sitdown_XXXX.npz

输出: E:\老人跌倒\data\custom_sitdown\
  keypoints=(n_frames, 33, 4)  [x, y, z, visibility]
  category=1 (SitDown)
"""
import os, time
import numpy as np
import cv2
import mediapipe as mp

# ─── 配置 ───
VIDEO = r"F:\动作数据集\50个坐下的动作.mp4"
CLASS_ID = 1               # SitDown
PREFIX = "sitdown"
OUTPUT_DIR = r"E:\老人跌倒\data\custom_sitdown"
LOG_FILE = r"E:\老人跌倒\data\extraction_sitdown_log.txt"

SKIP_FRAMES = 1          # 不跳帧
POSE_CHUNK = 300         # 每 N 帧重建 Pose，防状态退化
MIN_DETECTION_CONF = 0.4
MIN_TRACKING_CONF = 0.4

# ─── 分段参数（与 standup_wakeup v2 一致）───
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


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    log(f"\n{'='*60}")
    log(f"  Video: {os.path.basename(video_path)}")
    log(f"  Frames: {total_frames}, FPS: {fps:.1f}")

    frames = []
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


def movement_between(k1, k2):
    valid = (k1[:, 3] > 0.5) & (k2[:, 3] > 0.5)
    if valid.sum() < 10:
        return 0.0
    return float(np.linalg.norm(k2[valid, :2] - k1[valid, :2], axis=1).mean())


def segment_frames(frames):
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
        moves = [0.0]
        for i in range(1, len(seg)):
            moves.append(movement_between(seg[i-1][1], seg[i][1]))
        moves = np.array(moves)

        still = moves < STILL_MOVEMENT
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

    final_segments = [s for s in final_segments if len(s) >= MIN_SEG_FRAMES]
    return final_segments


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
            source_video=os.path.basename(VIDEO),
        )
        saved += 1
    return saved


if __name__ == "__main__":
    log(f"{'='*60}")
    log(f"  SitDown 分段提取 (50个坐下的动作.mp4)")
    log(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  SKIP={SKIP_FRAMES}, GAP={GAP_THRESHOLD}, STILL={STILL_MOVEMENT}/{STILL_WINDOW}")
    log(f"{'='*60}")

    if not os.path.exists(VIDEO):
        log(f"  ⚠ NOT FOUND: {VIDEO}")
    else:
        frames, fps = process_video(VIDEO)
        segments = segment_frames(frames)
        n = save_segments(segments, CLASS_ID, PREFIX)
        log(f"\n{'='*60}")
        log(f"  COMPLETE: {n} 个独立动作段保存到 {OUTPUT_DIR}")
        log(f"{'='*60}")
