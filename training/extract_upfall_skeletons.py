#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_upfall_skeletons.py
从 UP-Fall 3D 骨架数据集提取关键点，转为项目 .npz 格式

UP-Fall 数据说明:
- 33 关节 × (X, Y, Z) 三维坐标 (Kinect v1, 米制)
- 100 帧/序列，每个序列约 20-40% 是真正的摔倒帧
- LABEL 列: 0=摔倒, 1=非摔倒
- 活动 A1-A5 均为摔倒变体 (前摔/后摔/侧摔/坐摔)

输出格式 (与现有 pipeline 一致):
- keypoints: (N, 33, 4) float32  (x, y, z, visibility)
- labels: (N,) int32             (per-frame category index)
- category: int                   (主要类别)
- source: str                     (来源标识)
"""

import os, sys, zipfile, json
import numpy as np

# ── 配置 ──
UPFALL_DIR = r"F:\动作数据集\数据集（1）\3D+skeletons+UP-Fall+Dataset"
OUTPUT_DIR = r"E:\老人跌倒\data\upfall_features"

# UP-Fall 活动 → 我们的 6 分类映射
# 0=Fall, 1=SitDown, 2=StandUp, 3=Walking, 4=WakeUp, 5=Standing
UPFALL_ACTIVITY_MAP = {
    # A1: Falling forward using hands  -> Fall
    # A2: Falling forward using knees  -> Fall
    # A3: Falling backwards            -> Fall
    # A4: Falling sideways             -> Fall
    # A5: Falling sitting              -> Fall (更像 SitDown 但本质是摔倒)
    1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
}

# 坐标归一化: UP-Fall 用米制 (Kinect), MediaPipe 用 [0,1] 归一化
# 本脚本会将 X,Y 归一化到 [0,1]，保持 Z 不变（Z 很小，不会造成问题）
NORMALIZE_COORDS = True

# 是否也保存非摔倒帧（用于 Standing/Walking 训练）
# 非摔倒帧来自摔倒序列的前后部分，标签设为 5 (Standing)
INCLUDE_NONFALL = True
NONFALL_CATEGORY = 5  # Standing

# 摔倒帧提取策略
# "labeled": 只取 LABEL==0 的帧
# "all_as_fall": 整个摔倒序列全标为 Fall（不考虑 per-frame label）
FALL_STRATEGY = "labeled"


def parse_upfall_csv(csv_path):
    """
    解析 UP-Fall CSV 文件
    返回: (keypoints, labels) 
      keypoints: (N, 33, 3) — raw XYZ
      labels: (N,) — 0=fall, 1=non-fall
    """
    with open(csv_path, 'r') as f:
        content = f.read().strip()
    
    lines = content.split('\n')
    if not lines:
        return None, None
    
    header = lines[0]
    # 确认列数: 33 joints × 3 coords + LABEL = 100 列
    cols = header.split(',')
    expected_cols = 33 * 3 + 1  # 100
    if len(cols) != expected_cols:
        print(f"  [WARN] 列数异常: {len(cols)} (期望 {expected_cols})")
    
    frames = []
    labels = []
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) < expected_cols:
            continue
        
        try:
            values = [float(v.strip() or '0') for v in parts]
        except ValueError:
            continue
        
        # 前 99 个值 = 33 joints × 3 coords
        joints = np.array(values[:99], dtype=np.float32).reshape(33, 3)
        label = int(values[99])  # 0=fall, 1=non-fall
        
        frames.append(joints)
        labels.append(label)
    
    if not frames:
        return None, None
    
    return np.stack(frames), np.array(labels, dtype=np.int32)


def normalize_keypoints(kp_xyz):
    """
    将米制 XYZ 坐标归一化到 [0,1] 范围 (类似 MediaPipe 输出)
    
    策略: 对整个序列取 X/Y 的 min/max，线性映射到 [0,1]
    处理异常值: 使用 1st/99th 百分位裁剪
    """
    N = kp_xyz.shape[0]
    normalized = np.zeros((N, 33, 4), dtype=np.float32)
    
    x = kp_xyz[:, :, 0]
    y = kp_xyz[:, :, 1]
    z = kp_xyz[:, :, 2]
    
    # 使用百分位裁剪避免极端值影响归一化
    x_lo, x_hi = np.percentile(x[x > -10], 1), np.percentile(x[x < 10], 99)
    y_lo, y_hi = np.percentile(y[y > -10], 1), np.percentile(y[y < 10], 99)
    
    x_range = max(x_hi - x_lo, 0.01)
    y_range = max(y_hi - y_lo, 0.01)
    
    # 归一化 X, Y
    normalized[:, :, 0] = (x - x_lo) / x_range
    normalized[:, :, 1] = (y - y_lo) / y_range
    
    # Z 保持原值（Kinect Z 深度变化很小）
    normalized[:, :, 2] = z
    
    # Visibility: UP-Fall 所有关节点都是 Kinect 追踪到的，设为 1.0
    normalized[:, :, 3] = 1.0
    
    # Clip 到合理范围
    normalized[:, :, 0] = np.clip(normalized[:, :, 0], -0.5, 1.5)
    normalized[:, :, 1] = np.clip(normalized[:, :, 1], -0.5, 1.5)
    
    return normalized


def extract_subject(zip_path, subject_id, output_dir, stats):
    """提取单个被试的所有序列"""
    zf = zipfile.ZipFile(zip_path, 'r')
    csv_names = sorted(zf.namelist())
    
    saved_files = []
    total_fall_frames = 0
    total_nonfall_frames = 0
    
    for csv_name in csv_names:
        # 解析命名: C{cam}S{sub}A{act}T{trial}.csv
        # 提取原始数据
        raw = zf.read(csv_name)
        tmp_path = os.path.join(output_dir, '_temp.csv')
        with open(tmp_path, 'wb') as f:
            f.write(raw)
        
        kp_xyz, labels = parse_upfall_csv(tmp_path)
        os.remove(tmp_path)
        
        if kp_xyz is None:
            print(f"  [SKIP] {csv_name}: 无法解析")
            continue
        
        # 解析活动编号
        base = csv_name.replace('.csv', '').replace('_', '')
        act_match = None
        for char_idx, ch in enumerate(base):
            if ch == 'A' and char_idx + 1 < len(base) and base[char_idx + 1].isdigit():
                act_match = base[char_idx:char_idx + 2]
                break
        activity = int(act_match[1]) if act_match else 0
        
        # 归一化
        kp_normalized = normalize_keypoints(kp_xyz)
        
        # 策略: 按 per-frame LABEL 分类
        fall_mask = (labels == 0)
        nonfall_mask = (labels == 1)
        
        n_fall = fall_mask.sum()
        n_nonfall = nonfall_mask.sum()
        
        if n_fall == 0:
            print(f"  [SKIP] {csv_name}: 无摔倒帧 ({n_nonfall} 非摔倒)")
            continue
        
        # ── 分段策略: 把连续的摔倒帧和非摔倒帧分开 ──
        # 找连续段
        segments = []  # [(start_idx, end_idx, is_fall)]
        i = 0
        while i < len(labels):
            is_fall = (labels[i] == 0)
            j = i
            while j < len(labels) and (labels[j] == 0) == is_fall:
                j += 1
            if j - i >= 3:  # 至少 3 帧才保留
                segments.append((i, j, is_fall))
            i = j
        
        # 保存段
        for seg_idx, (start, end, is_fall) in enumerate(segments):
            seg_kp = kp_normalized[start:end]
            seg_labels = labels[start:end]
            
            if is_fall:
                category = 0  # Fall
                seg_tag = "Fall"
                total_fall_frames += (end - start)
            else:
                if not INCLUDE_NONFALL:
                    continue
                # 非摔倒帧在摔倒序列中 → 可能是 Standing / Walking
                # 简化处理: 标记为 Standing
                category = NONFALL_CATEGORY
                seg_tag = "Standing"
                total_nonfall_frames += (end - start)
            
            # 生成文件名
            source_id = csv_name.replace('.csv', '').replace('_', '-')
            fname = f"UPFALL_S{subject_id}_{source_id}_seg{seg_idx}_{seg_tag}.npz"
            save_path = os.path.join(output_dir, fname)
            
            np.savez_compressed(
                save_path,
                keypoints=seg_kp,
                labels=np.full(end - start, category, dtype=np.int32),
                category=np.int64(category),
                source=np.str_(fname.replace('.npz', ''))
            )
            saved_files.append(save_path)
    
    return saved_files, total_fall_frames, total_nonfall_frames


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    zip_files = sorted([
        f for f in os.listdir(UPFALL_DIR)
        if f.endswith('.zip')
    ])
    
    print(f"=" * 60)
    print(f"UP-Fall 骨架数据提取")
    print(f"=" * 60)
    print(f"源目录: {UPFALL_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"找到 {len(zip_files)} 个 zip: {zip_files}")
    print(f"策略: {FALL_STRATEGY}")
    print(f"坐标归一化: {NORMALIZE_COORDS}")
    print(f"包含非摔倒帧: {INCLUDE_NONFALL} (类别: {NONFALL_CATEGORY})")
    print()
    
    # 清空旧输出
    for old in os.listdir(OUTPUT_DIR):
        if old.endswith('.npz') or old == '_temp.csv':
            os.remove(os.path.join(OUTPUT_DIR, old))
    
    all_files = []
    grand_fall = 0
    grand_nonfall = 0
    
    for zf_name in zip_files:
        zip_path = os.path.join(UPFALL_DIR, zf_name)
        # 提取被试编号
        subj_id = zf_name.replace('SUBJECT', '').replace('.zip', '')
        
        print(f"\n--- {zf_name} (Subject {subj_id}) ---")
        files, nf, nnf = extract_subject(zip_path, subj_id, OUTPUT_DIR, {})
        all_files.extend(files)
        grand_fall += nf
        grand_nonfall += nnf
        print(f"  -> {len(files)} 段, Fall: {nf} 帧, Standing: {nnf} 帧")
    
    # ── 汇总 ──
    final_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npz')])
    
    print(f"\n{'=' * 60}")
    print(f"提取完成")
    print(f"{'=' * 60}")
    print(f"输出文件: {len(final_files)} 个 .npz")
    print(f"摔倒帧: {grand_fall}")
    print(f"非摔倒帧 (Standing): {grand_nonfall}")
    print(f"总帧数: {grand_fall + grand_nonfall}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # 统计各类别
    cat_counts = {}
    total_frames = 0
    for f in final_files:
        d = np.load(os.path.join(OUTPUT_DIR, f), allow_pickle=True)
        cat = int(d['category'])
        n = d['keypoints'].shape[0]
        cat_counts[cat] = cat_counts.get(cat, 0) + n
        total_frames += n
    
    print(f"\n类别分布:")
    cat_names = {0: 'Fall', 1: 'SitDown', 2: 'StandUp', 3: 'Walking', 4: 'WakeUp', 5: 'Standing'}
    for cat in sorted(cat_counts.keys()):
        print(f"  {cat} ({cat_names.get(cat, '?')}): {cat_counts[cat]} 帧")
    print(f"  总计: {total_frames} 帧")
    
    # 保存提取报告
    report = {
        "source": "UP-Fall 3D Skeleton Dataset",
        "zip_files": zip_files,
        "output_dir": OUTPUT_DIR,
        "num_files": len(final_files),
        "total_frames": total_frames,
        "fall_frames": grand_fall,
        "nonfall_frames": grand_nonfall,
        "categories": {str(k): v for k, v in cat_counts.items()},
        "settings": {
            "normalize_coords": NORMALIZE_COORDS,
            "fall_strategy": FALL_STRATEGY,
            "include_nonfall": INCLUDE_NONFALL,
            "nonfall_category": NONFALL_CATEGORY,
        }
    }
    report_path = os.path.join(OUTPUT_DIR, "extraction_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告: {report_path}")


if __name__ == "__main__":
    main()
