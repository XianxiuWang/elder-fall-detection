#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_fall_videos_v5.py — 分段 Pose 重建版
策略: 每 200 帧关闭并重建 MediaPipe Pose，绕开长时间运行的内存/状态问题
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
SKIP_FRAMES = 5
POSE_CHUNK = 200  # 每 200 帧重建 Pose
LOG_FILE = r"E:\老人跌倒\data\extraction_log_v5.txt"

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
        f.flush()
        os.fsync(f.fileno())

def process_video(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target = total // (SKIP_FRAMES + 1)
    
    log(f"\n{'='*60}")
    log(f"  视频: {base_name}  ({w}x{h}, {target} 目标帧, 每 {POSE_CHUNK} 帧重建 Pose)")
    log(f"{'='*60}")
    
    all_kpts = []
    frame_count = 0
    processed = 0
    detected = 0
    errors = 0
    t0 = time.time()
    
    def make_pose():
        return mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.3, min_tracking_confidence=0.3
        )
    
    pose = make_pose()
    chunk_start = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            if frame_count % (SKIP_FRAMES + 1) != 0: continue
            
            processed += 1
            
            # 每 POSE_CHUNK 帧重建 Pose
            if processed - chunk_start >= POSE_CHUNK:
                pose.close()
                pose = make_pose()
                chunk_start = processed
                log(f"  [POSE RESET] at frame {frame_count} (processed {processed})")
            
            try:
                scale = 640 / w
                small = cv2.resize(frame, (640, int(h * scale)))
                img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                results = pose.process(img_rgb)
                
                if results.pose_landmarks:
                    lm = np.zeros((33, 4), dtype=np.float32)
                    for j, lm_i in enumerate(results.pose_landmarks.landmark):
                        lm[j] = [lm_i.x, lm_i.y, lm_i.z, lm_i.visibility]
                    all_kpts.append(lm)
                    detected += 1
                else:
                    all_kpts.append(np.zeros((33, 4), dtype=np.float32))
            except Exception as e:
                errors += 1
                all_kpts.append(np.zeros((33, 4), dtype=np.float32))
                if errors <= 5:
                    log(f"  [WARN] frame {frame_count}: {e}")
            
            if processed % 300 == 0:
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 0.001)
                det_pct = detected / max(processed, 1) * 100
                log(f"  {processed}/{target} ({100*processed//max(target,1)}%) | {rate:.1f}fps | 检出:{det_pct:.1f}% | 错:{errors}")
    
    finally:
        pose.close()
        cap.release()
    
    # 保存为 .npz
    kp = np.array(all_kpts, dtype=np.float32)
    n = kp.shape[0]
    labels = np.full(n, 0, dtype=np.int32)
    fname = f"C2_{base_name}_seg00.npz"
    path = os.path.join(OUTPUT_DIR, fname)
    np.savez_compressed(path, keypoints=kp, labels=labels, category=0,
                        source=fname.replace('.npz', ''))
    size_kb = os.path.getsize(path) / 1024
    
    elapsed = time.time() - t0
    det_pct = detected / max(processed, 1) * 100
    log(f"    -> {fname}  ({n} 帧, {size_kb:.0f} KB)")
    log(f"  [OK] {processed}帧 ({elapsed:.0f}s, {processed/max(elapsed,0.001):.1f}fps) | 检出:{det_pct:.1f}% | 错误:{errors}")
    return [path]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log(f"Extraction v5 started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 清理旧 Fall 数据
    existing = os.listdir(OUTPUT_DIR)
    keywords = ["Ways to Fall", "50_Ways_to_Fall50", "50个摔倒的动作"]
    removed = 0
    for old in existing:
        if old.endswith('.npz') and any(kw in old for kw in keywords):
            os.remove(os.path.join(OUTPUT_DIR, old))
            removed += 1
    log(f"清理旧数据: {removed} 个")
    
    # 恢复 WakeUp
    import shutil
    src = r"E:\老人跌倒\data\custom_5class\C4_WakeUp_50_Ways_to_Wake_Up50种睡醒方式.npz"
    dst = os.path.join(OUTPUT_DIR, "C4_WakeUp_50_Ways_to_Wake_Up50种睡醒方式.npz")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        log("[RESTORE] WakeUp")
    
    # 逐个处理
    all_files = []
    for v in VIDEOS:
        if not os.path.exists(v):
            log(f"[SKIP] 文件不存在: {v}")
            continue
        try:
            files = process_video(v)
            all_files.extend(files)
        except Exception as e:
            log(f"[FAIL] {os.path.basename(v)}: {e}")
    
    # 总结
    final = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npz')]
    total_frames = 0
    for f in final:
        try:
            d = np.load(os.path.join(OUTPUT_DIR, f), allow_pickle=True)
            total_frames += d['keypoints'].shape[0]
        except: pass
    
    log(f"\n{'='*60}")
    log(f"  完成! 新增: {len(all_files)} 段, 总计: {len(final)} npz, {total_frames} 帧")
    log(f"  下一步: python -u training/train_6class_v2.py")
    log(f"{'='*60}")

if __name__ == "__main__":
    main()
