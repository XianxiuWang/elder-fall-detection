#!/usr/bin/env python3
"""
prepare_6class.py — 准备六分类训练数据
========================================
从现有五分类数据中提取/生成 "Standing/Idle" (C5) 样本，
然后合并为六分类数据集。

策略:
  1. 从 Walking 序列开头提取站立帧（走路前的静止）
  2. 从 StandUp 序列开头提取站立帧
  3. 从 WakeUp 序列末尾提取静止帧
  4. 合成高斯噪声站立样本（补充数据量）

用法:
    python prepare_6class.py
    python prepare_6class.py --min-frames 500 --noise-count 200
"""

import os
import sys
import argparse
import numpy as np

_DATA_DIR = r"E:\老人跌倒\data"
_SRC_DIR = os.path.join(_DATA_DIR, "custom_5class")
_OUT_DIR = os.path.join(_DATA_DIR, "custom_6class")
_EXISTING_CLASSES = {
    "C0": "Fall",
    "C1": "SitDown",
    "C2": "StandUp",
    "C3": "Walking",
    "C4": "WakeUp",
}


def extract_standing_frames(npz_path: str, class_prefix: str) -> list:
    """
    从 .npz 文件中提取站立/静止帧。
    规则：取序列前后 5% 中变异系数最低的窗口。
    """
    data = np.load(npz_path, allow_pickle=True)
    kpts = data['keypoints']  # (N, 33, 4) or (N, 33, 3)
    if kpts.ndim == 3 and kpts.shape[2] >= 3:
        kpts_3d = kpts[:, :, :3]
    else:
        kpts_3d = kpts

    n_frames = kpts_3d.shape[0]
    if n_frames < 10:
        return []

    # 取开头和结尾 10% 的帧（过渡区域可能不典型）
    head_n = max(3, n_frames // 10)
    tail_n = max(3, n_frames // 10)
    candidate_idxs = list(range(head_n)) + list(range(n_frames - tail_n, n_frames))

    # 窗口大小 30（2秒 @ 15fps）
    window = 30
    best_frames = []

    for start in range(0, n_frames - window, window // 2):
        seg = kpts_3d[start:start + window]
        if seg.shape[0] < 10:
            continue
        # 稳定性评估：所有关键点的帧间位移标准差
        diffs = np.diff(seg, axis=0)
        stability = -np.std(diffs)  # 越大越稳定（负值越小）
        hip_y = seg[:, 23:25, 1].mean(axis=1)   # 髋部 Y 坐标
        hip_stability = -np.std(hip_y)           # 髋部垂直稳定度
        # 必须是站立高度（髋部在膝盖之上）
        knee_y = seg[:, 25:27, 1].mean(axis=1)
        if np.mean(hip_y) > np.mean(knee_y) * 0.95:
            continue  # 髋部太低，可能坐着或躺着

        best_frames.append((stability + hip_stability * 2, start, window))

    best_frames.sort(reverse=True)
    result = []
    used_ranges = []

    for score, start, w in best_frames[:5]:  # 取前5个最稳定窗口
        end = min(start + w, n_frames)
        # 避免重叠
        overlap = False
        for us, ue in used_ranges:
            if max(start, us) < min(end, ue):
                overlap = True
                break
        if overlap:
            continue

        seg = kpts[start:end]
        result.append(seg)
        used_ranges.append((start, end))

    return result


def generate_synthetic_standing(ref_kpts: np.ndarray, n_samples: int) -> np.ndarray:
    """
    基于参考站立帧合成带噪声的站立样本。
    模拟自然站立时 ±3° 的躯干轻微摇摆。
    """
    n_frames_per_sample = 30
    # 取参考帧的均值姿态
    if ref_kpts.ndim == 3:
        mean_pose = ref_kpts.mean(axis=0)  # (33, 4)
    else:
        mean_pose = ref_kpts.reshape(-1, 4) if ref_kpts.shape[-1] == 4 else ref_kpts

    synthetic = np.zeros((n_samples * n_frames_per_sample, 33, 4), dtype=np.float32)
    for i in range(n_samples * n_frames_per_sample):
        # 添加高斯噪声模拟自然摇摆：躯干部分 σ=0.008，四肢 σ=0.003
        noise = np.zeros((33, 4), dtype=np.float32)
        # 肩部和髋部：较大噪声
        noise[11:13] = np.random.normal(0, 0.008, (2, 4))
        noise[23:25] = np.random.normal(0, 0.008, (2, 4))
        # 头部
        noise[0:5] = np.random.normal(0, 0.005, (5, 4))
        # 四肢
        noise[13:23] = np.random.normal(0, 0.003, (10, 4))
        noise[25:33] = np.random.normal(0, 0.003, (8, 4))
        # 可见度保持高
        noise[:, 3] = np.clip(np.random.normal(0, 0.05, 33), -0.1, 0.1)
        frame = mean_pose + noise
        frame[:, :2] = np.clip(frame[:, :2], 0, 1)
        frame[:, 2] = np.clip(frame[:, 2], -0.5, 0.5)
        frame[:, 3] = np.clip(frame[:, 3], 0, 1)
        synthetic[i] = frame
    return synthetic


def main():
    parser = argparse.ArgumentParser(description="Prepare 6-class training data")
    parser.add_argument("--min-frames", type=int, default=500,
                        help="Minimum standing frames to extract")
    parser.add_argument("--noise-count", type=int, default=100,
                        help="Number of synthetic standing samples (×30 frames each)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be extracted without saving")
    args = parser.parse_args()

    os.makedirs(_OUT_DIR, exist_ok=True)

    # Step 1: Copy existing 5-class data
    print("=" * 60)
    print("Step 1: Copying existing 5-class data")
    existing_files = [f for f in os.listdir(_SRC_DIR) if f.endswith('.npz')]
    for f in existing_files:
        src = os.path.join(_SRC_DIR, f)
        dst = os.path.join(_OUT_DIR, f)
        if not args.dry_run and not os.path.exists(dst):
            import shutil
            shutil.copy2(src, dst)

    # Step 2: Extract standing frames from existing sequences
    print(f"\nStep 2: Extracting standing/idle frames from {len(existing_files)} files")
    standing_segments = []
    total_extracted = 0

    for fname in existing_files:
        path = os.path.join(_SRC_DIR, fname)
        prefix = fname[:2]
        if prefix not in _EXISTING_CLASSES:
            continue

        # 只从 Walking(C3), StandUp(C2), WakeUp(C4) 中提取
        if prefix not in ("C2", "C3", "C4"):
            continue

        segments = extract_standing_frames(path, prefix)
        for seg in segments:
            standing_segments.append(seg)
            total_extracted += seg.shape[0]
            if total_extracted >= args.min_frames:
                break
        if total_extracted >= args.min_frames:
            break

    print(f"  Extracted {len(standing_segments)} segments, {total_extracted} frames")

    # Step 3: Generate synthetic standing data
    print(f"\nStep 3: Generating {args.noise_count} synthetic standing samples")
    if standing_segments:
        # 使用第一个提取的站立帧作为参考
        ref = standing_segments[0]
        synthetic = generate_synthetic_standing(ref, args.noise_count)
        synth_frames = synthetic.shape[0]
    else:
        # 回退：用一个默认站立姿态
        default_pose = np.zeros((33, 4), dtype=np.float32)
        default_pose[0] = [0.5, 0.1, -0.1, 0.99]    # 鼻
        default_pose[7] = [0.43, 0.45, 0.0, 0.99]   # 左耳
        default_pose[8] = [0.57, 0.45, 0.0, 0.99]    # 右耳
        default_pose[11] = [0.40, 0.55, -0.1, 0.99]  # 左肩
        default_pose[12] = [0.60, 0.55, -0.1, 0.99]  # 右肩
        default_pose[13] = [0.35, 0.67, 0.0, 0.99]   # 左肘
        default_pose[14] = [0.65, 0.67, 0.0, 0.99]   # 右肘
        default_pose[15] = [0.33, 0.80, 0.1, 0.99]   # 左腕
        default_pose[16] = [0.67, 0.80, 0.1, 0.99]   # 右腕
        default_pose[23] = [0.42, 0.70, 0.0, 0.99]   # 左髋
        default_pose[24] = [0.58, 0.70, 0.0, 0.99]   # 右髋
        default_pose[25] = [0.40, 0.85, 0.1, 0.99]   # 左膝
        default_pose[26] = [0.60, 0.85, 0.1, 0.99]   # 右膝
        default_pose[27] = [0.38, 0.95, 0.15, 0.99]  # 左踝
        default_pose[28] = [0.62, 0.95, 0.15, 0.99]  # 右踝
        ref_arr = default_pose.reshape(1, 33, 4)
        synthetic = generate_synthetic_standing(ref_arr, args.noise_count)
        synth_frames = synthetic.shape[0]
        print("  (using default standing pose, no extracted data available)")

    print(f"  Generated {synth_frames} synthetic standing frames")

    # Step 4: Save C5_Standing data
    print(f"\nStep 4: Saving C5_Standing data")
    if args.dry_run:
        print("  [DRY RUN] Would save to", _OUT_DIR)
        return

    # Split into .npz files (each ~1000 frames to avoid huge files)
    frames_per_file = 900
    n_files = max(1, synth_frames // frames_per_file)
    for i in range(n_files):
        start = i * frames_per_file
        end = min(start + frames_per_file, synth_frames)
        seg = synthetic[start:end]
        labels = np.full(seg.shape[0], 5, dtype=np.int32)  # class 5 = Standing
        fpath = os.path.join(_OUT_DIR, f"C5_Standing_synthetic_{i+1:02d}.npz")
        np.savez_compressed(fpath, keypoints=seg, labels=labels, category=5, source="synthetic")
        print(f"  Saved {fpath} ({seg.shape[0]} frames)")

    # Also save extracted real standing frames if available
    if standing_segments:
        all_extracted = np.concatenate(standing_segments, axis=0)
        labels = np.full(all_extracted.shape[0], 5, dtype=np.int32)
        fpath = os.path.join(_OUT_DIR, "C5_Standing_extracted.npz")
        np.savez_compressed(fpath, keypoints=all_extracted.astype(np.float32),
                            labels=labels, category=5, source="extracted")
        print(f"  Saved {fpath} ({all_extracted.shape[0]} frames)")

    print(f"\n  Total C5 files: {n_files + (1 if standing_segments else 0)}")
    print(f"  Output directory: {_OUT_DIR}")
    print(f"\n{'='*60}")
    print("DONE. Ready for 6-class retraining.")
    print(f"Update train_fall_classifier.py CLASS_NAMES to:")
    print(f"  ['Fall', 'SitDown', 'StandUp', 'Walking', 'WakeUp', 'Standing']")
    print(f"Then run: python -m training.train_fall_classifier --data {_OUT_DIR}")


if __name__ == "__main__":
    main()
