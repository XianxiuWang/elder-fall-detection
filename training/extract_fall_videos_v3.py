#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_fall_videos.py v3 — 使用 OpenCV VideoCapture + 帧级异常恢复
"""

import os, sys, time, shutil
import numpy as np
import cv2
import mediapipe as mp

VIDEOS = [
    r"F:\动作数据集\50 Ways to Fall.mp4",
    r"F:\动作数据集\50 Ways to Fall50种摔倒方式.mp4",
    r"F:\动作数据集\50个摔倒的动作.mp4",
]
OUTPUT_DIR = r"E:\老人跌倒\data\custom_6class"
CATEGORY = 0  # Fall

mp_pose = mp.solutions.pose

LOG_PATH = os.path.join(os.path.dirname(OUTPUT_DIR), "extraction_log_v3.txt")

def log(msg):
    msg = str(msg)
    print(msg, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')
        f.flush()
        os.fsync(f.fileno())


def extract_video(video_path, log_prefix="  "):
    base = os.path.splitext(os.path.basename(video_path))[0]
    
    # check if already done
    existing = os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []
    if any(base in f and f.endswith('.npz') for f in existing):
        log(f"{log_prefix}[SKIP] {base} — already extracted")
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"{log_prefix}[FAIL] Cannot open: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    log(f"{log_prefix}{base}: {total_frames} frames @ {fps:.0f} fps")
    
    saved_files = []
    keypoints_list = []
    seg_idx = 0
    processed = 0
    detected = 0
    failed = 0
    
    t0 = time.time()
    pose = None
    
    try:
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        
        last_log = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed += 1
            
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    lm = np.zeros((33, 4), dtype=np.float32)
                    for j, lm_obj in enumerate(results.pose_landmarks.landmark):
                        lm[j] = [lm_obj.x, lm_obj.y, lm_obj.z, lm_obj.visibility]
                    keypoints_list.append(lm)
                    detected += 1
                else:
                    keypoints_list.append(np.zeros((33, 4), dtype=np.float32))
            except Exception as e:
                failed += 1
                keypoints_list.append(np.zeros((33, 4), dtype=np.float32))
                if failed <= 3:
                    log(f"{log_prefix}[WARN] frame {processed}: {e}")
            
            # progress every 200 frames
            if processed - last_log >= 200:
                last_log = processed
                elapsed = time.time() - t0
                pct = processed / max(total_frames, 1) * 100
                fps_proc = processed / max(elapsed, 0.001)
                det_rate = detected / max(processed, 1) * 100
                log(f"{log_prefix}{processed}/{total_frames} ({pct:.0f}%) | "
                    f"{fps_proc:.1f}fps | det:{det_rate:.1f}% | fail:{failed}")
            
            # segment save every 3000 frames
            if len(keypoints_list) >= 3000:
                kp = np.array(keypoints_list, dtype=np.float32)
                n = kp.shape[0]
                labels = np.full(n, CATEGORY, dtype=np.int32)
                fname = f"C2_{base}_seg{seg_idx:02d}.npz"
                path = os.path.join(OUTPUT_DIR, fname)
                np.savez_compressed(path, keypoints=kp, labels=labels, category=CATEGORY,
                                    source=fname.replace('.npz', ''))
                size_kb = os.path.getsize(path) / 1024
                log(f"{log_prefix}  -> {fname} ({n}f, {size_kb:.0f}KB)")
                saved_files.append(path)
                keypoints_list = []
                seg_idx += 1
        
        # save remaining
        if keypoints_list:
            kp = np.array(keypoints_list, dtype=np.float32)
            n = kp.shape[0]
            labels = np.full(n, CATEGORY, dtype=np.int32)
            fname = f"C2_{base}_seg{seg_idx:02d}.npz"
            path = os.path.join(OUTPUT_DIR, fname)
            np.savez_compressed(path, keypoints=kp, labels=labels, category=CATEGORY,
                                source=fname.replace('.npz', ''))
            size_kb = os.path.getsize(path) / 1024
            log(f"{log_prefix}  -> {fname} ({n}f, {size_kb:.0f}KB)")
            saved_files.append(path)
    
    finally:
        if pose is not None:
            pose.close()
        cap.release()
    
    elapsed = time.time() - t0
    det_rate = detected / max(processed, 1) * 100
    log(f"{log_prefix}[OK] {processed}f ({elapsed:.0f}s) | det:{det_rate:.1f}% | fail:{failed}")
    return saved_files


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # clear old log
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(f"=== Extraction v3: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    existing = set(f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npz'))
    log(f"Existing npz: {len(existing)}")
    
    # Clean old partial extractions from these videos
    for old in list(existing):
        if any(kw in old for kw in ["Ways to Fall50", "50个摔倒的动作"]):
            os.remove(os.path.join(OUTPUT_DIR, old))
            log(f"  [DEL] {old}")
            existing.discard(old)
    
    all_saved = []
    for v in VIDEOS:
        if not os.path.exists(v):
            log(f"[SKIP] Missing: {v}")
            continue
        saved = extract_video(v)
        all_saved.extend(saved)
    
    # Restore WakeUp file if missing
    wakeup_src = r"E:\老人跌倒\data\custom_5class\C4_WakeUp_50_Ways_to_Wake_Up50种睡醒方式.npz"
    wakeup_dst = os.path.join(OUTPUT_DIR, "C4_WakeUp_50_Ways_to_Wake_Up50种睡醒方式.npz")
    if os.path.exists(wakeup_src) and not os.path.exists(wakeup_dst):
        shutil.copy2(wakeup_src, wakeup_dst)
        log(f"[RESTORE] WakeUp file restored from custom_5class")
    
    final = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npz')]
    total_frames = 0
    for f in final:
        try:
            d = np.load(os.path.join(OUTPUT_DIR, f), allow_pickle=True)
            total_frames += d['keypoints'].shape[0]
        except:
            pass
    
    log(f"\n{'='*60}")
    log(f"DONE! New segments: {len(all_saved)}, Total npz: {len(final)}, Frames: {total_frames}")
    log(f"Next: python -u training/train_6class_v2.py")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
