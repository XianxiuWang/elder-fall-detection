#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_fall_videos_v4.py — 皮实版摔倒视频关键点提取
改进:
  1. OpenCV 直读视频（不用 ffmpeg 解帧中介）
  2. MediaPipe static_image_mode=False + smooth 追踪（更快更稳）
  3. 帧级 try/except 保护
  4. 每段立即落盘
"""

import os, sys, time
import numpy as np
import mediapipe as mp
import cv2

VIDEOS = [
    r"F:\动作数据集\50 Ways to Fall.mp4",
    r"F:\动作数据集\50 Ways to Fall50种摔倒方式.mp4",
    r"F:\动作数据集\50个摔倒的动作.mp4",
]
OUTPUT_DIR = r"E:\老人跌倒\data\custom_6class"

SKIP_FRAMES = 5       # 每 6 帧取 1 帧 (~5fps from 30fps)
SEGMENT_SIZE = 3000   # 每段最多帧数
LOG_FILE = r"E:\老人跌倒\data\extraction_log_v4.txt"

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
        f.flush()
        os.fsync(f.fileno())

def process_video(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"  [FAIL] 无法打开: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 目标提取帧数
    target_frames = total_frames // (SKIP_FRAMES + 1)
    log(f"\n{'='*60}")
    log(f"  视频: {base_name}")
    log(f"  {w}x{h}, {fps:.1f}fps, {total_frames} 总帧 → 跳过每 {SKIP_FRAMES} 帧 → ~{target_frames} 目标帧")
    log(f"{'='*60}")
    
    saved_files = []
    keypoints_list = []
    segment_idx = 0
    frame_count = 0
    processed = 0
    detected = 0
    errors = 0
    
    t0 = time.time()
    
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 跳帧
            if frame_count % (SKIP_FRAMES + 1) != 0:
                continue
            
            processed += 1
            
            try:
                # 缩放到 640 宽
                scale = 640 / w
                small = cv2.resize(frame, (640, int(h * scale)))
                img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                
                results = pose.process(img_rgb)
                
                if results.pose_landmarks:
                    lm = np.zeros((33, 4), dtype=np.float32)
                    for j, landmark in enumerate(results.pose_landmarks.landmark):
                        lm[j] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
                    keypoints_list.append(lm)
                    detected += 1
                else:
                    keypoints_list.append(np.zeros((33, 4), dtype=np.float32))
                    
            except Exception as e:
                errors += 1
                keypoints_list.append(np.zeros((33, 4), dtype=np.float32))
                if errors <= 5:
                    log(f"  [WARN] 帧 {frame_count} 异常: {e}")
            
            # 日志
            if processed % 300 == 0:
                elapsed = time.time() - t0
                fps_proc = processed / max(elapsed, 0.001)
                det_rate = detected / max(processed, 1) * 100
                log(f"  {processed}/{target_frames} ({processed * 100 // max(target_frames, 1)}%) | "
                    f"{fps_proc:.1f} fps | 检出: {det_rate:.1f}% | 错误: {errors}")
            
            # 分段保存
            if len(keypoints_list) >= SEGMENT_SIZE:
                fp = save_segment(keypoints_list, base_name, segment_idx)
                if fp:
                    saved_files.append(fp)
                keypoints_list = []
                segment_idx += 1
        
        # 保存剩余
        if keypoints_list:
            fp = save_segment(keypoints_list, base_name, segment_idx)
            if fp:
                saved_files.append(fp)
        
    finally:
        pose.close()
        cap.release()
    
    elapsed = time.time() - t0
    det_rate = detected / max(processed, 1) * 100
    log(f"  [OK] {processed} 帧完成 ({elapsed:.0f}s, {processed/max(elapsed,0.001):.1f} fps) | "
        f"检出: {det_rate:.1f}% | 错误: {errors}")
    
    return saved_files


def save_segment(kp_list, base_name, seg_idx):
    kp = np.array(kp_list, dtype=np.float32)
    n = kp.shape[0]
    labels = np.full(n, 0, dtype=np.int32)  # category 0 = Fall
    
    fname = f"C2_{base_name}_seg{seg_idx:02d}.npz"
    path = os.path.join(OUTPUT_DIR, fname)
    np.savez_compressed(path, keypoints=kp, labels=labels, category=0,
                        source=fname.replace('.npz', ''))
    size_kb = os.path.getsize(path) / 1024
    log(f"    -> {fname}  ({n} 帧, {size_kb:.0f} KB)")
    return path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log(f"Extraction v4 started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 清理旧的 Fall 提取数据
    existing = set(f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npz'))
    fall_keywords = ["Ways to Fall", "50 Ways to Fall50", "50个摔倒的动作"]
    removed = 0
    for old in list(existing):
        if any(kw in old for kw in fall_keywords):
            os.remove(os.path.join(OUTPUT_DIR, old))
            removed += 1
    log(f"现有: {len(existing)} 个 .npz, 清理旧 Fall 数据: {removed} 个")
    
    # 确保 WakeUp 文件存在（从 custom_5class 恢复）
    wakeup_src = r"E:\老人跌倒\data\custom_5class\C4_WakeUp_50_Ways_to_Wake_Up50种睡醒方式.npz"
    wakeup_dst = os.path.join(OUTPUT_DIR, "C4_WakeUp_50_Ways_to_Wake_Up50种睡醒方式.npz")
    if os.path.exists(wakeup_src) and not os.path.exists(wakeup_dst):
        import shutil
        shutil.copy2(wakeup_src, wakeup_dst)
        log(f"  [RESTORE] WakeUp 文件已恢复")
    
    # 处理所有视频
    all_files = []
    for v in VIDEOS:
        if not os.path.exists(v):
            log(f"  [SKIP] 文件不存在: {v}")
            continue
        files = process_video(v)
        all_files.extend(files)
    
    # 总结
    final = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npz')]
    total_frames = 0
    for f in final:
        try:
            d = np.load(os.path.join(OUTPUT_DIR, f), allow_pickle=True)
            total_frames += d['keypoints'].shape[0]
        except:
            pass
    
    log(f"\n{'='*60}")
    log(f"  完成! 新增: {len(all_files)} 段, 总计: {len(final)} 个 .npz")
    log(f"  总帧数: {total_frames}")
    log(f"\n  下一步: python -u training/train_6class_v2.py")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
