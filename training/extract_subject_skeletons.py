#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_subject_skeletons.py v2
从 Subject.1~9 的 PNG 帧中提取 MediaPipe 33 关键点

优化: Pose 对象全运行期间只初始化一次
"""

import os, sys, time, json
import numpy as np
import cv2
import mediapipe as mp

# ── 配置 ──
SUBJECT_DIR = r"F:\动作数据集\数据集（1）"
OUTPUT_DIR = r"E:\老人跌倒\data\subject_features"

ACTION_MAP = {
    "Fall backwards":   0,
    "Fall forward":     0,
    "Fall left":        0,
    "Fall right":       0,
    "Fall sitting":     0,
    "Sit down":         1,
    "Walk":             3,
}

MAX_WIDTH = 640
BATCH_SIZE = 500
MIN_DETECTION_CONF = 0.3
SUBJECTS = list(range(1, 10))

_flush_counter = [0]


def imread_unicode(path):
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is not None and img.shape[1] > MAX_WIDTH:
        h, w = img.shape[:2]
        scale = MAX_WIDTH / w
        img = cv2.resize(img, (MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def flush_buffer(kp_buf, lbl_buf, category, subj_id):
    if not kp_buf:
        return []
    kp = np.array(kp_buf, dtype=np.float32)
    labels = np.array(lbl_buf, dtype=np.int32)
    n = kp.shape[0]
    cat_names = {0: "Fall", 1: "SitDown", 3: "Walking"}
    seg_idx = _flush_counter[0]
    _flush_counter[0] += 1
    fname = f"SUBJ_S{subj_id:02d}_{cat_names.get(category, 'X')}_seg{seg_idx:03d}.npz"
    path = os.path.join(OUTPUT_DIR, fname)
    np.savez_compressed(path, keypoints=kp, labels=labels,
                        category=np.int64(category),
                        source=np.str_(fname.replace('.npz', '')))
    size_kb = os.path.getsize(path) / 1024
    print(f"      -> {fname} ({n} 帧, {size_kb:.0f} KB)", flush=True)
    return [path]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"=" * 60, flush=True)
    print(f"Subject.1~9 MediaPipe 关键点提取 v2", flush=True)
    print(f"=" * 60, flush=True)
    print(f"被试: {SUBJECTS}", flush=True)
    print(f"动作: {list(ACTION_MAP.keys())}", flush=True)
    print(f"输出: {OUTPUT_DIR}", flush=True)
    
    # ═══ 一次性初始化 Pose ═══
    print(f"\n初始化 MediaPipe Pose...", flush=True)
    t_init = time.time()
    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,
        min_detection_confidence=MIN_DETECTION_CONF,
        min_tracking_confidence=MIN_DETECTION_CONF,
    )
    print(f"  就绪 ({time.time() - t_init:.1f}s)", flush=True)
    
    t_start = time.time()
    grand_stats = {"detected": 0, "missed": 0, "failed": 0, "total": 0}
    cat_counts = {}
    total_files = 0
    global_frame_idx = 0
    
    try:
        for sid in SUBJECTS:
            sp = os.path.join(SUBJECT_DIR, f"Subject.{sid}")
            if not os.path.isdir(sp):
                print(f"\nSubject.{sid}: 目录不存在", flush=True)
                continue
            
            # 收集所有 PNG
            print(f"\nSubject.{sid}: 扫描文件...", flush=True)
            all_images = []
            for action_name in sorted(os.listdir(sp)):
                ap = os.path.join(sp, action_name)
                if not os.path.isdir(ap) or action_name not in ACTION_MAP:
                    continue
                category = ACTION_MAP[action_name]
                pngs = sorted([f for f in os.listdir(ap) if f.lower().endswith('.png')])
                for png_name in pngs:
                    all_images.append((os.path.join(ap, png_name), category, action_name))
            
            if not all_images:
                print(f"  无可用图片", flush=True)
                continue
            
            total = len(all_images)
            print(f"  {total} 帧 | 开始推理...", flush=True)
            
            kp_buf, lbl_buf = [], []
            current_cat = None
            subj_detected = 0
            subj_missed = 0
            subj_failed = 0
            t_subj = time.time()
            
            for i, (img_path, category, action_name) in enumerate(all_images):
                global_frame_idx += 1
                grand_stats["total"] += 1
                
                # 进度
                if i % 100 == 0 and i > 0:
                    elapsed = time.time() - t_subj
                    fps = i / elapsed if elapsed > 0 else 0
                    det_rate = subj_detected / max(i, 1) * 100
                    eta = (total - i) / max(fps, 0.01)
                    print(f"    {i}/{total} | {fps:.1f} fps | "
                          f"检出:{det_rate:.0f}% | ETA:{eta:.0f}s", flush=True)
                
                try:
                    img = imread_unicode(img_path)
                    if img is None:
                        subj_failed += 1
                        grand_stats["failed"] += 1
                        continue
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = pose.process(img_rgb)
                    
                    if results.pose_landmarks:
                        lm = np.zeros((33, 4), dtype=np.float32)
                        for j, landmark in enumerate(results.pose_landmarks.landmark):
                            lm[j] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
                        kp_buf.append(lm)
                        lbl_buf.append(category)
                        subj_detected += 1
                        grand_stats["detected"] += 1
                        current_cat = category
                        
                        if len(kp_buf) >= BATCH_SIZE:
                            cat_counts[current_cat] = cat_counts.get(current_cat, 0) + len(kp_buf)
                            total_files += len(flush_buffer(kp_buf, lbl_buf, current_cat, sid))
                            kp_buf, lbl_buf = [], []
                    else:
                        subj_missed += 1
                        grand_stats["missed"] += 1
                        
                except Exception as e:
                    subj_failed += 1
                    grand_stats["failed"] += 1
                    if subj_failed <= 3:
                        print(f"    [WARN] {os.path.basename(img_path)}: {e}", flush=True)
            
            # 保存剩余
            if kp_buf and current_cat is not None:
                cat_counts[current_cat] = cat_counts.get(current_cat, 0) + len(kp_buf)
                total_files += len(flush_buffer(kp_buf, lbl_buf, current_cat, sid))
            
            elapsed = time.time() - t_subj
            det_rate = subj_detected / max(total, 1) * 100
            print(f"    完成 {elapsed:.0f}s | 检出:{det_rate:.1f}% | "
                  f"丢失:{subj_missed} | 失败:{subj_failed}", flush=True)
    
    finally:
        pose.close()
    
    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 60}", flush=True)
    print(f"提取完成 (总耗时: {total_elapsed:.0f}s / {total_elapsed/60:.1f}min)", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"输出文件: {total_files} 个 .npz", flush=True)
    print(f"有效帧: {grand_stats['detected']} / {grand_stats['total']} "
          f"({grand_stats['detected']/max(grand_stats['total'],1)*100:.1f}%)", flush=True)
    
    cat_names = {0: "Fall", 1: "SitDown", 2: "StandUp", 3: "Walking", 4: "WakeUp", 5: "Standing"}
    for cat in sorted(cat_counts.keys()):
        print(f"  {cat} ({cat_names.get(cat, '?')}): {cat_counts[cat]} 帧", flush=True)
    print(f"  总计: {sum(cat_counts.values())} 帧", flush=True)
    print(f"  丢失: {grand_stats['missed']} | 失败: {grand_stats['failed']}", flush=True)
    
    report = {
        "source": "Subject.1~9 PNG + MediaPipe Pose",
        "num_files": total_files,
        "stats": grand_stats,
        "categories": {str(k): v for k, v in cat_counts.items()},
        "elapsed_seconds": total_elapsed,
    }
    with open(os.path.join(OUTPUT_DIR, "extraction_report.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告: {os.path.join(OUTPUT_DIR, 'extraction_report.json')}", flush=True)


if __name__ == "__main__":
    main()
