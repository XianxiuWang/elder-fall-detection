"""
hard_negative_mining.py — v5: MediaPipe Video Mode, 直接构造窗口
关键优化: 用 Pose(video mode) 顺序处理视频段，自动追踪
"""
import os, sys, json, time
import numpy as np
import cv2
import mediapipe as mp

VIDEO_PATH = r"F:\动作数据集\100 Ways to Walk100种走路方式.mp4"
RESULT_JSON = r"D:\Users\wangxianxiu\clawd\results_100 Ways to Walk100种走路方式.json"
OUTPUT_DIR = r"E:\老人跌倒\data\custom_6class"
WINDOW_SIZE = 30
SKIP_FRAMES = 3

sys.stdout.reconfigure(encoding='utf-8')

# ── Step 1: 找难例 ──
print("[1/3] Loading results...")
with open(RESULT_JSON, 'r', encoding='utf-8') as f:
    results = json.load(f)
fps = results['fps']
raw_falls = [r for r in results['predictions'] if r['raw'] == 'Fall']
print(f"  Raw Fall windows: {len(raw_falls)}")

# 计算每个窗口需要的帧范围
target_windows = []  # [(start_frame, end_frame, center_frame)]
for rw in raw_falls:
    fc = rw['frame']
    start = fc - 29 * SKIP_FRAMES  # fc - 87
    end = fc
    if start >= 0:
        target_windows.append((start, end, fc))

# 合并重叠的帧范围
if not target_windows:
    print("  No valid windows!"); sys.exit(0)

target_windows.sort()
merged_ranges = []
for s, e, fc in target_windows:
    if not merged_ranges or s > merged_ranges[-1]['end'] + SKIP_FRAMES * 3:
        merged_ranges.append({'start': s, 'end': e, 'centers': [(s, e, fc)]})
    else:
        merged_ranges[-1]['end'] = max(merged_ranges[-1]['end'], e)
        merged_ranges[-1]['centers'].append((s, e, fc))

print(f"  Target windows: {len(target_windows)}, Merged ranges: {len(merged_ranges)}")

# ── Step 2: 对每段跑 MediaPipe Video Mode ──
print(f"\n[2/3] MediaPipe Pose (video mode) on {len(merged_ranges)} segments...")
t0 = time.time()

kpts_by_frame = {}  # frame_idx -> (33, 4) array

for ri, rng in enumerate(merged_ranges):
    s, e = rng['start'], rng['end']
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, s - 10))  # 前跳几个做缓冲
    
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,       # Video mode = track across frames
        model_complexity=1,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4)
    
    frame_idx = max(0, s - 10)
    while frame_idx <= e:
        ret, frame = cap.read()
        if not ret:
            break
        if s <= frame_idx <= e:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                kpts = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                                 for lm in result.pose_landmarks.landmark], dtype=np.float32)
                kpts_by_frame[frame_idx] = kpts
        frame_idx += 1
    
    pose.close()
    cap.release()
    
    elapsed = time.time() - t0
    n_found = sum(1 for f in range(s, e+1) if f in kpts_by_frame)
    duration = (e - s) / fps
    print(f"  Seg {ri+1}/{len(merged_ranges)}: frames {s}-{e} ({duration:.1f}s), "
          f"got {n_found}/{e-s+1} poses, {elapsed:.0f}s")

total_elapsed = time.time() - t0
print(f"  Total: {len(kpts_by_frame)} poses in {total_elapsed:.0f}s")

# ── Step 3: 构建窗口 + 保存 ──
print(f"\n[3/3] Building & saving {len(target_windows)} window samples...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
saved = 0

for wi, (s, e, fc) in enumerate(target_windows):
    kpt_seq = []
    valid = 0
    for f_idx in range(s, e + 1, SKIP_FRAMES):
        if f_idx in kpts_by_frame:
            kpt_seq.append(kpts_by_frame[f_idx])
            valid += 1
        else:
            kpt_seq.append(np.zeros((33, 4), dtype=np.float32))
    
    if valid >= 15:
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f"C3_Walking_hardneg_{wi:03d}.npz"),
            keypoints=np.array(kpt_seq, dtype=np.float32),
            labels=np.full(WINDOW_SIZE, 3, dtype=np.int32),
            category=np.int64(3),
            source=np.str_(f"100WaysWalk_hardneg_w{wi}_{fc}_at{fc/fps:.0f}s"))
        saved += 1

print(f"\n{'='*60}")
print(f"  DONE: {saved}/{len(target_windows)} hard negative samples saved")
print(f"  Total time: {total_elapsed:.0f}s")
print(f"  Output: {OUTPUT_DIR}")
print(f"{'='*60}")
