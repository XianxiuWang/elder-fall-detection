#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_fall_videos.py v2 — 从摔倒视频提取 MediaPipe 关键点
修复: 帧级超时、异常跳过、日志频繁刷新、ffmpeg 解帧方案
"""

import os, sys, time, json, subprocess, signal, tempfile, shutil
import numpy as np
import mediapipe as mp
import cv2

# ── 输入视频 ──
VIDEOS = [
    r"F:\动作数据集\50 Ways to Fall.mp4",
    r"F:\动作数据集\50 Ways to Fall50种摔倒方式.mp4",
    r"F:\动作数据集\50个摔倒的动作.mp4",
]
OUTPUT_DIR = r"E:\老人跌倒\data\custom_6class"
CATEGORY = 0  # Fall

mp_pose = mp.solutions.pose

# ── 查找 ffmpeg ──
def _find_ffmpeg():
    """查找 imageio-ffmpeg 的 ffmpeg 二进制"""
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

FFMPEG = _find_ffmpeg()
print(f"ffmpeg: {FFMPEG}", flush=True)


class TimeoutError(Exception):
    pass


def handler(signum, frame):
    raise TimeoutError("Frame processing timeout")


def extract_video_ffmpeg(video_path, out_dir, log_fn, max_frames_per_segment=3000):
    """
    方案: ffmpeg 一次性解所有帧到临时目录 → MediaPipe 逐帧推理
    好处: 不会因单帧卡死整个管线, 进度可追踪
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    tmp_dir = os.path.join(tempfile.gettempdir(), f"fall_extract_{os.getpid()}")
    os.makedirs(tmp_dir, exist_ok=True)

    # ── Step 1: ffmpeg 解帧 ──
    log_fn(f"\n{'='*60}")
    log_fn(f"  视频: {base_name}")
    log_fn(f"  Step 1: ffmpeg 解帧 -> {tmp_dir}")
    log_fn(f"{'='*60}")

    # 获取视频总帧数 (用 ffmpeg 的 ffprobe; 没有 ffprobe 则跳过用 OpenCV)
    ffprobe_exe = FFMPEG.replace('ffmpeg', 'ffprobe')
    total_frames = 0
    if os.path.exists(ffprobe_exe):
        probe_cmd = [
            ffprobe_exe, '-v', 'error', '-select_streams', 'v:0',
            '-count_packets', '-show_entries', 'stream=nb_read_packets',
            '-of', 'csv=p=0', video_path
        ]
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            total_frames = int(result.stdout.strip() or 0)
        except:
            pass
    if total_frames <= 0:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

    log_fn(f"  总帧数: {total_frames}")

    # ffmpeg 解帧到临时目录 (fps=5, 256p 宽)
    extract_cmd = [
        FFMPEG, '-y', '-i', video_path,
        '-vf', 'fps=5,scale=640:-1',
        '-q:v', '2',
        os.path.join(tmp_dir, 'frame_%06d.jpg')
    ]
    log_fn(f"  ffmpeg: {' '.join(extract_cmd)}")
    try:
        result = subprocess.run(extract_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log_fn(f"  [WARN] ffmpeg stderr: {result.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        log_fn(f"  [WARN] ffmpeg 超时, 继续处理已有帧")

    # 列出解出的帧
    frame_files = sorted([
        f for f in os.listdir(tmp_dir)
        if f.endswith('.jpg') or f.endswith('.png')
    ])
    total_frames = len(frame_files)
    log_fn(f"  解帧完成: {total_frames} 帧")

    if total_frames == 0:
        log_fn(f"  [FAIL] 解帧失败, 回退到 OpenCV 方案")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return extract_video_opencv(video_path, out_dir, log_fn, max_frames_per_segment)

    # ── Step 2: MediaPipe 推理 ──
    log_fn(f"  Step 2: MediaPipe 关键点提取 ({total_frames} 帧)")
    
    saved_files = []
    keypoints_list = []
    segment_idx = 0
    detected = 0
    failed = 0
    skipped = 0
    
    t0 = time.time()
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    
    try:
        for i, fname in enumerate(frame_files):
            if i % 300 == 0 and i > 0:
                elapsed = time.time() - t0
                pct = i / total_frames * 100
                fps_proc = i / elapsed if elapsed > 0 else 0
                det_rate = detected / max(i - skipped, 1) * 100
                log_fn(f"  {i}/{total_frames} ({pct:.0f}%) | "
                       f"{fps_proc:.1f} fps | 检出: {det_rate:.1f}% | 失败: {failed}")

            filepath = os.path.join(tmp_dir, fname)
            
            try:
                # 读图片
                img = cv2.imread(filepath)
                if img is None:
                    skipped += 1
                    keypoints_list.append(np.zeros((33, 4), dtype=np.float32))
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # MediaPipe 推理（带帧级 try/except）
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
                failed += 1
                keypoints_list.append(np.zeros((33, 4), dtype=np.float32))
                if failed <= 5:
                    log_fn(f"  [WARN] 帧 {i} 失败: {e}")

            # 分段保存
            if len(keypoints_list) >= max_frames_per_segment:
                saved_files.append(
                    _save_segment(keypoints_list, out_dir, base_name, segment_idx, CATEGORY, log_fn)
                )
                keypoints_list = []
                segment_idx += 1

        # 保存剩余帧
        if keypoints_list:
            saved_files.append(
                _save_segment(keypoints_list, out_dir, base_name, segment_idx, CATEGORY, log_fn)
            )

    finally:
        pose.close()
        # 清理临时目录
        shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.time() - t0
    valid_frames = total_frames - skipped
    det_rate = detected / max(valid_frames, 1) * 100
    log_fn(f"  [OK] {total_frames} 帧完成 ({elapsed:.0f}s) | "
           f"检出: {det_rate:.1f}% | 跳过: {skipped} | 失败: {failed}")
    return saved_files


def extract_video_opencv(video_path, out_dir, log_fn, max_frames_per_segment=3000):
    """OpenCV 回退方案"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_fn(f"  [FAIL] 无法打开: {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log_fn(f"  OpenCV 模式: {total_frames} 帧")

    saved_files = []
    keypoints_list = []
    segment_idx = 0
    frame_count = 0
    detected = 0
    failed = 0
    
    t0 = time.time()
    pose = mp_pose.Pose(
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
            
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                if results.pose_landmarks:
                    lm = np.zeros((33, 4), dtype=np.float32)
                    for j, landmark in enumerate(results.pose_landmarks.landmark):
                        lm[j] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
                    keypoints_list.append(lm)
                    detected += 1
                else:
                    keypoints_list.append(np.zeros((33, 4), dtype=np.float32))
            except Exception as e:
                failed += 1
                keypoints_list.append(np.zeros((33, 4), dtype=np.float32))
                if failed <= 3:
                    log_fn(f"  [WARN] 帧 {frame_count} 处理失败: {e}")
            
            if frame_count % 300 == 0:
                elapsed = time.time() - t0
                pct = frame_count / max(total_frames, 1) * 100
                fps_proc = frame_count / max(elapsed, 0.001)
                det_rate = detected / max(frame_count, 1) * 100
                log_fn(f"  {frame_count}/{total_frames} ({pct:.0f}%) | "
                       f"{fps_proc:.1f} fps | 检出: {det_rate:.1f}%")
            
            if len(keypoints_list) >= max_frames_per_segment:
                saved_files.append(
                    _save_segment(keypoints_list, out_dir, base_name, segment_idx, CATEGORY, log_fn)
                )
                keypoints_list = []
                segment_idx += 1
        
        if keypoints_list:
            saved_files.append(
                _save_segment(keypoints_list, out_dir, base_name, segment_idx, CATEGORY, log_fn)
            )
    finally:
        pose.close()
    
    cap.release()
    elapsed = time.time() - t0
    det_rate = detected / max(frame_count, 1) * 100
    log_fn(f"  [OK] OpenCV: {frame_count}帧 ({elapsed:.0f}s) | 检出:{det_rate:.1f}% | 失败:{failed}")
    return saved_files


def _save_segment(keypoints_list, out_dir, base_name, seg_idx, category, log_fn):
    """保存一段关键点为 .npz"""
    kp = np.array(keypoints_list, dtype=np.float32)
    n = kp.shape[0]
    labels = np.full(n, category, dtype=np.int32)
    
    fname = f"C2_{base_name}_seg{seg_idx:02d}.npz"
    path = os.path.join(out_dir, fname)
    np.savez_compressed(path, keypoints=kp, labels=labels, category=category,
                        source=fname.replace('.npz', ''))
    size_kb = os.path.getsize(path) / 1024
    log_fn(f"    -> {fname}  ({n} 帧, {size_kb:.0f} KB)")
    return path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log_file = os.path.join(os.path.dirname(OUTPUT_DIR), "extraction_log_v2.txt")
    
    def log(msg):
        print(msg, flush=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
            f.flush()
            os.fsync(f.fileno())
    
    log(f"Extraction v2 started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ── 列出已有文件 ──
    existing = set(f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npz'))
    log(f"现有数据: {len(existing)} 个 .npz")
    
    # ⚠️ 修复: 只删除名含 "Ways to Fall" / "50种摔倒" / "50个摔倒" 的 Fall 文件,
    #    不删 WakeUp 文件（虽然名含 "50_Ways_to_Wake"）
    fall_keywords = ["Ways to Fall", "50 Ways to Fall50", "50个摔倒的动作"]
    for old in list(existing):
        should_delete = any(kw in old for kw in fall_keywords)
        if should_delete:
            old_path = os.path.join(OUTPUT_DIR, old)
            os.remove(old_path)
            log(f"  [DEL] {old}")
            existing.discard(old)
    
    # ── 提取 ──
    all_files = []
    for v in VIDEOS:
        if not os.path.exists(v):
            log(f"  [SKIP] 文件不存在: {v}")
            continue
        files = extract_video_ffmpeg(v, OUTPUT_DIR, log)
        all_files.extend(files)
    
    # ── 恢复 WakeUp 文件（如果被误删）──
    wakeup_src = r"E:\老人跌倒\data\custom_5class\C4_WakeUp_50_Ways_to_Wake_Up50种睡醒方式.npz"
    wakeup_dst = os.path.join(OUTPUT_DIR, "C4_WakeUp_50_Ways_to_Wake_Up50种睡醒方式.npz")
    if os.path.exists(wakeup_src) and not os.path.exists(wakeup_dst):
        shutil.copy2(wakeup_src, wakeup_dst)
        log(f"  [RESTORE] WakeUp 文件已从 custom_5class 恢复")
    
    # ── 验证 ──
    final = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npz')]
    log(f"\n{'='*60}")
    log(f"  完成! 新增: {len(all_files)} 段, 总计: {len(final)} 个 .npz")
    
    total_frames = 0
    for f in final:
        try:
            d = np.load(os.path.join(OUTPUT_DIR, f), allow_pickle=True)
            total_frames += d['keypoints'].shape[0]
        except:
            pass
    log(f"  总帧数: {total_frames}")
    log(f"  日志: {log_file}")
    log(f"\n  下一步: python -u training/train_6class_v2.py")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
