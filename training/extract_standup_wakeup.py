#!/usr/bin/env python3
"""
extract_standup_wakeup.py — 从 50 Ways 视频提取 StandUp＋WakeUp 关键点
==================================================================
视频:
  StandUp: "50 Ways to Stand -50种站立方式.mp4"   → class 2
  WakeUp:  "50 Ways to Wake Up50种睡醒方式.mp4"    → class 4

策略:
  - 分段重建 MediaPipe Pose（每 200 帧重置，防止内存/状态退化）
  - 跳过静帧（关节位移 < 阈值），保留实际动作片段
  - 每段 200 帧保存为 1 个 .npz，含 30 帧窗口的滑窗数据
  
输出: E:\老人跌倒\data\custom_6class\
  文件命名: standup_XXXX.npz / wakeup_XXXX.npz
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
OUTPUT_DIR = r"E:\老人跌倒\data\custom_6class"
SKIP_FRAMES = 6          # 每 N 帧处理一帧
POSE_CHUNK = 200          # 每 CHUNK 帧重建 Pose
WINDOW_SIZE = 30          # 滑窗大小
WINDOW_STRIDE = 6         # 滑窗步长
MIN_MOVEMENT = 0.008      # 最小关节位移（过滤静帧）
LOG_FILE = r"E:\老人跌倒\data\extraction_standup_wakeup_log.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


# ================================================================
# 分段 Pose 处理（每 POSE_CHUNK=200 帧重建，内存安全）
# ================================================================
def create_pose():
    """创建新的 MediaPipe Pose 实例"""
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def process_video(video_path, class_id, prefix):
    """处理一个长视频，提取全部有效帧的关键点"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    log(f"\n{'='*60}")
    log(f"  Video: {os.path.basename(video_path)}")
    log(f"  Class: {class_id} ({prefix})")
    log(f"  Frames: {total_frames}, FPS: {fps:.1f}")
    log(f"  Skip: {SKIP_FRAMES}, Chunk: {POSE_CHUNK}")
    
    all_keypoints = []      # [(frame_idx, kpts_33x3), ...]
    frame_idx = 0
    processed = 0
    frames_in_chunk = 0
    chunk_idx = 0
    t_start = time.time()
    
    pose = create_pose()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % SKIP_FRAMES != 0:
            frame_idx += 1
            continue
        
        # 每 POSE_CHUNK 帧重建 Pose（防止 MediaPipe 状态退化）
        if frames_in_chunk >= POSE_CHUNK:
            frames_in_chunk = 0
            chunk_idx += 1
            log(f"  ── chunk {chunk_idx} reset @ frame {frame_idx} ──")
            pose.close()
            pose = create_pose()
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            kpts = np.array([[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark],
                            dtype=np.float32)
            all_keypoints.append((frame_idx, kpts))
        
        processed += 1
        frames_in_chunk += 1
        frame_idx += 1
        
        # Progress
        if processed % 500 == 0:
            elapsed = time.time() - t_start
            pct = frame_idx / total_frames * 100
            log(f"    frame {frame_idx}/{total_frames} ({pct:.1f}%), "
                f"valid={len(all_keypoints)}, {processed/elapsed:.0f} f/s")
    
    pose.close()
    cap.release()
    elapsed = time.time() - t_start
    log(f"  Done: {processed} frames processed, {len(all_keypoints)} valid poses "
        f"({elapsed:.1f}s, {processed/elapsed:.0f} f/s)")
    
    return all_keypoints, fps


# ================================================================
# 过滤静帧 + 保存
# ================================================================
def compute_movement(kpts_list):
    """计算相邻帧之间的平均关节位移（kpts_list 是 (33, 3) 数组的列表）"""
    if len(kpts_list) < 2:
        return [0.0]
    movements = [0.0]
    for i in range(1, len(kpts_list)):
        prev = kpts_list[i-1]   # (33, 3)
        curr = kpts_list[i]     # (33, 3)
        valid = curr[:, 2] > 0.5
        if valid.sum() > 10:
            dist = np.linalg.norm(curr[valid, :2] - prev[valid, :2], axis=1).mean()
        else:
            dist = 0.0
        movements.append(dist)
    return movements


def save_segments(all_keypoints, class_id, prefix, fps):
    """
    保存连续关键点序列为 .npz（与现有 custom_6class 格式一致:
    keypoints=(n_frames, 33, 3), category=int）。
    每 500 帧保存一个文件。
    """
    # 检查是否已有文件（断点续传）
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"{prefix}_") and f.endswith('.npz')]
    if existing:
        log(f"  ⚠ {len(existing)} existing files found, clearing old data...")
        for f in existing:
            os.remove(os.path.join(OUTPUT_DIR, f))
    
    if len(all_keypoints) < WINDOW_SIZE:
        log(f"  ⚠ Only {len(all_keypoints)} valid frames (need {WINDOW_SIZE}+), skipping")
        return 0
    
    kpts_array = np.array([kp[1] for kp in all_keypoints], dtype=np.float32)
    n_frames = len(kpts_array)
    
    # 分段保存，每 500 帧一个文件
    frames_per_file = 500
    file_count = 0
    
    for seg_start in range(0, n_frames, frames_per_file):
        seg_end = min(seg_start + frames_per_file, n_frames)
        segment = kpts_array[seg_start:seg_end]
        
        # 跳过过短的段（< 窗口大小）
        if len(segment) < WINDOW_SIZE:
            continue
        
        out = os.path.join(OUTPUT_DIR, f"{prefix}_{file_count:04d}.npz")
        np.savez_compressed(
            out,
            keypoints=segment,
            category=class_id,
            n_frames=len(segment),
            source_video=os.path.basename(VIDEOS[0][0] if class_id == 2 else VIDEOS[1][0]),
        )
        file_count += 1
    
    windows_est = sum(
        max(0, (min(seg_s + frames_per_file, n_frames) - seg_s - WINDOW_SIZE) // WINDOW_STRIDE + 1)
        for seg_s in range(0, n_frames, frames_per_file)
    )
    
    log(f"  → {file_count} files ({n_frames} frames, ~{windows_est} windows "
        f"@ stride={WINDOW_STRIDE})")
    return file_count


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    log(f"{'='*60}")
    log(f"  StandUp + WakeUp Keypoint Extractor")
    log(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*60}")
    
    total_files = 0
    for video_path, class_id, prefix in VIDEOS:
        if not os.path.exists(video_path):
            log(f"  ⚠ NOT FOUND: {video_path}")
            continue
        
        all_keypoints, fps = process_video(video_path, class_id, prefix)
        n = save_segments(all_keypoints, class_id, prefix, fps)
        total_files += n
    
    log(f"\n{'='*60}")
    log(f"  COMPLETE: {total_files} files saved to {OUTPUT_DIR}")
    log(f"{'='*60}")
