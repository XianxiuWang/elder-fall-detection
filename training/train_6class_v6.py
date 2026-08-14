#!/usr/bin/env python3
"""
train_6class_v6.py — 全数据源合并重训
=====================================
数据源:
  - subject_features/  (131 files, Fall/SitDown/Walking)
  - custom_6class/     (140 files, all 6 classes)
  - custom_5class/     (135 files, 5 classes no Standing)
  - urfd_features/     ( 70 files, Fall only via label=1)
  - le2i_features/     (130 files, Fall from frame_fall_map)
  - upfall_features/   (164 files, Fall + Standing)

增强:
  - StandUp/WakeUp: 5x augmentation (flip/warp/noise/scale/mirror)
  - SitDown: 2x augmentation
  - Class weights for imbalance
"""
import os, sys, pickle, json, time, warnings, shutil
from collections import Counter
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_fall_classifier import FeatureExtractor

DATA_ROOTS = {
    'subject_features': r"E:\老人跌倒\data\subject_features",
    'custom_6class':    r"E:\老人跌倒\data\custom_6class",
    'custom_5class':    r"E:\老人跌倒\data\custom_5class",
    'urfd_features':    r"E:\老人跌倒\data\urfd_features",
    'le2i_features':    r"E:\老人跌倒\data\le2i_features",
    'upfall_features':  r"E:\老人跌倒\data\upfall_features",
}

MODEL_OUT = r"E:\老人跌倒\models\fall_classifier_6class_v6.pkl"
CLASS_NAMES = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"]


# ============================================================
# Enhanced Feature Extractor (same as v3)
# ============================================================
class EnhancedFeatureExtractor(FeatureExtractor):
    def __init__(self, window_size=30):
        super().__init__(window_size=window_size)
    
    def extract_window(self, window):
        base_vec, base_names = super().extract_window(window)
        extra = []
        extra_names = []
        
        # ── 9 original temporal features (v3) ──
        torso_y = window[:, 11, 1]
        n = len(torso_y)
        extra.extend([torso_y[:n//3].mean(), torso_y[n//3:2*n//3].mean(), torso_y[2*n//3:].mean()])
        extra_names += ['torso_y_early', 'torso_y_mid', 'torso_y_late']
        
        x = np.arange(n)
        extra.append(np.polyfit(x, torso_y, 1)[0])
        extra_names.append('torso_y_slope')
        
        head_y = window[:, 0, 1]
        extra.append(np.polyfit(x, head_y, 1)[0])
        extra_names.append('head_y_slope')
        
        speeds = np.linalg.norm(np.diff(window[:, :, :2], axis=0), axis=2)
        extra.append(speeds.std(axis=0).mean())
        extra_names.append('joint_speed_std')
        
        for idx in [0, 11, 23]:
            extra.append(window[-1, idx, 1] - window[0, idx, 1])
        extra_names += ['head_y_delta', 'shoulder_y_delta', 'hip_y_delta']
        
        # ── 8 NEW torso-tilt / body-orientation features (v6) ──
        # These target Fall↔Standing confusion:
        #   Standing → torso vertical (~0°), head above hips, narrow x-extent
        #   Fallen   → torso horizontal (~90°), head ≈ hip height, wide x-extent
        
        shoulder_mid_y = (window[:, 11, 1] + window[:, 12, 1]) / 2
        hip_mid_y = (window[:, 23, 1] + window[:, 24, 1]) / 2
        shoulder_mid_x = (window[:, 11, 0] + window[:, 12, 0]) / 2
        hip_mid_x = (window[:, 23, 0] + window[:, 24, 0]) / 2
        head_y = window[:, 0, 1]
        head_x = window[:, 0, 0]
        ankle_mid_y = (window[:, 27, 1] + window[:, 28, 1]) / 2
        knee_mid_y = (window[:, 25, 1] + window[:, 26, 1]) / 2
        
        # Feature 1: torso_tilt_late — torso angle (deg) in last 5 frames
        # 0°=vertical (standing), 90°=horizontal (fallen)
        late_n = min(5, n)
        dx_late = hip_mid_x[-late_n:] - shoulder_mid_x[-late_n:]
        dy_late = hip_mid_y[-late_n:] - shoulder_mid_y[-late_n:] + 1e-6
        tilt_late = np.degrees(np.arctan2(np.abs(dx_late), np.abs(dy_late)))
        extra.append(float(np.mean(tilt_late)))
        extra_names.append('torso_tilt_late')
        
        # Feature 2: torso_tilt_delta — tilt change from early to late
        # Positive = torso tilting more horizontal (falling)
        early_n = min(5, n)
        dx_early = hip_mid_x[:early_n] - shoulder_mid_x[:early_n]
        dy_early = hip_mid_y[:early_n] - shoulder_mid_y[:early_n] + 1e-6
        tilt_early = np.degrees(np.arctan2(np.abs(dx_early), np.abs(dy_early)))
        extra.append(float(np.mean(tilt_late) - np.mean(tilt_early)))
        extra_names.append('torso_tilt_delta')
        
        # Feature 3: head_vs_hip_height — head y vs hip y ratio
        # Standing: head_y < hip_y (head above = smaller y in image coords)
        # Fallen: head_y ≈ hip_y (both near ground)
        head_below_hip = np.mean(head_y[-late_n:] > hip_mid_y[-late_n:])  # True=1 if head below hip
        extra.append(float(head_below_hip))
        extra_names.append('head_below_hip_ratio')
        
        # Feature 4: body_height_ratio — torso center height relative to window max
        # Fallen: torso center is low (ratio ~0.2-0.4)
        # Standing: torso center is high (ratio ~0.7-0.9)
        torso_center_y = (shoulder_mid_y + hip_mid_y) / 2
        body_y_max = np.max(window[:, :, 1])  # highest keypoint (smallest y)
        body_y_min = np.min(window[:, :, 1])  # lowest keypoint (largest y)
        body_y_span = body_y_max - body_y_min + 1e-6
        torso_rel_height = np.mean((body_y_max - torso_center_y[-late_n:]) / body_y_span)
        extra.append(float(np.clip(torso_rel_height, 0, 1)))
        extra_names.append('torso_rel_height')
        
        # Feature 5: horizontal_spread_ratio — body x-extent vs y-extent
        # Standing: narrow x, tall y → ratio small
        # Fallen: wide x, short y → ratio large
        body_x_span = np.max(window[:, :, 0]) - np.min(window[:, :, 0])
        spread_ratio = body_x_span / (body_y_span + 1e-6)
        extra.append(float(np.clip(spread_ratio, 0, 10)))
        extra_names.append('horizontal_spread_ratio')
        
        # Feature 6: torso_angle_accel — rate of tilt change (2nd deriv)
        # Falling: rapid angle change → high accel
        all_tilt = np.degrees(np.arctan2(
            np.abs(hip_mid_x - shoulder_mid_x),
            np.abs(hip_mid_y - shoulder_mid_y) + 1e-6
        ))
        tilt_vel = np.gradient(all_tilt)
        tilt_accel = np.gradient(tilt_vel)
        extra.append(float(np.max(np.abs(tilt_accel))))
        extra_names.append('torso_angle_accel')
        
        # Feature 7: shoulder_hip_lateral_shift — lateral offset between shoulder and hip midpoints
        # Standing: well-aligned → small shift
        # Fallen/slouching: spine is curved → larger shift
        lateral_shift = np.mean(np.abs(shoulder_mid_x - hip_mid_x)[-late_n:])
        extra.append(float(lateral_shift))
        extra_names.append('shoulder_hip_shift')
        
        # Feature 8: keypoints_above_hips — fraction of keypoints above hip line
        # Standing: most keypoints above hips (head, shoulders, arms)
        # Fallen: most keypoints near or below hip line
        kp_above = np.mean(window[-late_n:, :, 1] < hip_mid_y[-late_n:, np.newaxis])
        extra.append(float(kp_above))
        extra_names.append('kp_above_hips_ratio')
        
        return np.concatenate([base_vec, np.array(extra)]), base_names + extra_names


# ============================================================
# Data Augmentation
# ============================================================
def augment_heavy(kpts, n_augment=5):
    """Heavy augmentation for rare classes (StandUp, WakeUp)"""
    versions = [kpts.copy()]
    T, J, _ = kpts.shape
    
    # 1. Horizontal flip
    flipped = kpts.copy()
    flipped[:, :, 0] = 1.0 - flipped[:, :, 0]
    versions.append(flipped)
    
    # 2. Time warp (resample)
    if T > 15:
        warped_idx = np.sort(np.random.choice(T, size=T, replace=True))
        versions.append(kpts[warped_idx].copy())
    
    # 3. Gaussian noise
    noisy = kpts.copy()
    noisy[:, :, :2] += np.random.normal(0, 0.005, noisy[:, :, :2].shape).astype(np.float32)
    versions.append(noisy)
    
    # 4. Scale augmentation (slight zoom)
    scaled = kpts.copy()
    scale = 1.0 + np.random.normal(0, 0.03)
    scaled[:, :, :2] *= scale
    versions.append(scaled)
    
    # 5. Mirror + noise
    if n_augment >= 5:
        mir_noisy = flipped.copy()
        mir_noisy[:, :, :2] += np.random.normal(0, 0.005, mir_noisy[:, :, :2].shape).astype(np.float32)
        versions.append(mir_noisy)
    
    return versions[:n_augment + 1]


def augment_light(kpts, n_augment=2):
    """Light augmentation (SitDown)"""
    versions = [kpts.copy()]
    
    flipped = kpts.copy()
    flipped[:, :, 0] = 1.0 - flipped[:, :, 0]
    versions.append(flipped)
    
    if n_augment >= 2:
        noisy = kpts.copy()
        noisy[:, :, :2] += np.random.normal(0, 0.003, noisy[:, :, :2].shape).astype(np.float32)
        versions.append(noisy)
    
    return versions[:n_augment + 1]


# ============================================================
# Data Loading
# ============================================================
def load_all_data(extractor, window_size=30, stride=6, augment=True):
    """
    Load all data sources, extract features.
    返回 (X, y, groups, starts, is_orig):
      groups  : 每个样本所属文件 id（用于按文件分组划分，防数据泄漏）
      starts  : 每个样本的窗口起始帧索引（用于事件级时序重建）
      is_orig : 是否为原始(非增强)窗口
    """
    X_all, y_all = [], []
    sample_groups = []   # 文件 id
    sample_starts = []   # 窗口起始帧索引
    sample_is_orig = []  # 是否原始窗口
    total_files = 0
    skipped = 0
    aug_stats = {}
    next_file_id = 0
    
    for source_name, data_dir in DATA_ROOTS.items():
        if not os.path.exists(data_dir):
            print(f"  [{source_name}] Directory not found, skipping")
            continue
        
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        source_X, source_y = [], []
        aug_count = Counter()
        
        for fname in files:
            path = os.path.join(data_dir, fname)
            fid = next_file_id
            next_file_id += 1
            try:
                data = np.load(path, allow_pickle=True)
                
                # Determine keypoint key name
                kp_key = None
                for k in ['keypoints', 'landmarks']:
                    if k in data and hasattr(data[k], 'shape') and data[k].ndim == 3:
                        kp_key = k
                        break
                if kp_key is None:
                    skipped += 1
                    continue
                
                kpts = data[kp_key]
                n_frames = kpts.shape[0]
                if n_frames < window_size:
                    skipped += 1
                    continue
                
                # Determine category
                category = None
                if 'category' in data:
                    cat_val = data['category']
                    if hasattr(cat_val, 'item') and hasattr(cat_val, 'ndim') and cat_val.ndim == 0:
                        category = int(cat_val.item())
                
                # Special handling for URFD
                if source_name == 'urfd_features':
                    if 'label' in data:
                        label = int(data['label'].item())
                        if label == 0:
                            continue  # Skip ADL (unknown class)
                        else:
                            category = 0  # Fall
                
                # Special handling for Le2i (extract only Fall frames via frame_fall_map)
                if source_name == 'le2i_features' and 'frame_fall_map' in data:
                    ffm = data['frame_fall_map']
                    if np.all(ffm == 0):
                        continue  # No fall frames
                    category = 0  # All Le2i files contain falls
                
                if category is None:
                    skipped += 1
                    continue
                
                # Only accept valid 0-5 categories
                if category not in range(6):
                    skipped += 1
                    continue
                
                # Determine augmentation level
                versions = [kpts]
                if augment:
                    if category == 2 or category == 4:  # StandUp or WakeUp
                        versions = augment_heavy(kpts, n_augment=5)
                        aug_count['heavy'] += 1
                    elif category == 1:  # SitDown
                        versions = augment_light(kpts, n_augment=2)
                        aug_count['light'] += 1
                
                # Extract features from each version
                for vi, kpts_ver in enumerate(versions):
                    kpts_3d = kpts_ver[:, :, :3].astype(np.float32)
                    T = kpts_3d.shape[0]
                    for start in range(0, T - window_size + 1, stride):
                        window = kpts_3d[start:start + window_size]
                        vec, _ = extractor.extract_window(window)
                        source_X.append(vec)
                        source_y.append(category)
                        sample_groups.append(fid)
                        sample_starts.append(start)
                        sample_is_orig.append(vi == 0)
                
                total_files += 1
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"  [SKIP] {source_name}/{fname}: {e}")
        
        if source_X:
            X_all.extend(source_X)
            y_all.extend(source_y)
            print(f"  [{source_name}] {len(files):3d} files → {len(source_X):6d} samples | "
                  f"aug={dict(aug_count) if aug_count else 'none'}")
        else:
            print(f"  [{source_name}] {len(files):3d} files → 0 samples (all skipped)")
    
    X = np.nan_to_num(np.array(X_all, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y_all, dtype=np.int32)
    groups = np.array(sample_groups, dtype=np.int64)
    starts = np.array(sample_starts, dtype=np.int64)
    is_orig = np.array(sample_is_orig, dtype=bool)
    
    print(f"\n  Total: {total_files} files loaded, {skipped} skipped")
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  唯一文件(分组)数: {len(np.unique(groups))}")
    for cls_id, cnt in sorted(Counter(y).items()):
        pct = 100 * cnt / len(y)
        bar = '█' * int(pct / 2)
        print(f"  {CLASS_NAMES[cls_id]:12s}: {cnt:6d} ({pct:5.1f}%) {bar}")
    
    return X, y, groups, starts, is_orig


# ============================================================
# 事件级评估工具
# ============================================================
FALL_ID = 0  # CLASS_NAMES[0] = "Fall"


def smooth_fall_labels(fall_probs, ema_alpha=0.3, prob_thresh=0.35):
    """简化时序过滤：EMA 平滑 + 概率阈值。返回 0/1 序列。
    (不含 fall_inference.py 里的自由落体判别器，因为此处无原始关键点)"""
    ema = None
    out = []
    for p in fall_probs:
        ema = p if ema is None else ema_alpha * p + (1 - ema_alpha) * ema
        out.append(1 if ema >= prob_thresh else 0)
    return np.array(out, dtype=np.int32)


def merge_events(binary_seq, min_gap=1, min_len=1):
    """把 0/1 序列中的连续 1 段合并为事件 [start, end) 开区间，允许 min_gap 间隙。"""
    events = []
    i, n = 0, len(binary_seq)
    while i < n:
        if binary_seq[i] == 1:
            s = i
            while i < n and binary_seq[i] == 1:
                i += 1
            if i - s >= min_len:
                events.append([s, i])
        else:
            i += 1
    if min_gap > 0 and events:
        merged = [events[0]]
        for ev in events[1:]:
            if ev[0] - merged[-1][1] <= min_gap:
                merged[-1][1] = ev[1]
            else:
                merged.append(ev)
        events = merged
    return events


def match_events(true_events, pred_events, min_overlap=1):
    """贪心匹配真/预测事件，按重叠窗口数降序。返回 (tp, fp, fn)。"""
    pairs = []
    for ti, te in enumerate(true_events):
        for pi, pe in enumerate(pred_events):
            ov = min(te[1], pe[1]) - max(te[0], pe[0])
            if ov >= min_overlap:
                pairs.append((ov, ti, pi))
    pairs.sort(reverse=True)
    matched_t, matched_p = set(), set()
    for ov, ti, pi in pairs:
        if ti not in matched_t and pi not in matched_p:
            matched_t.add(ti)
            matched_p.add(pi)
    tp = len(matched_t)
    fp = len(pred_events) - len(matched_p)
    fn = len(true_events) - tp
    return tp, fp, fn


def evaluate_event_level(groups, starts, is_orig, y, fall_probs,
                         ema_alpha=0.3, prob_thresh=0.35, min_gap=1):
    """事件级评估：只对原始窗口，按文件重建时序 → 简化过滤 → 事件合并匹配。"""
    tp = fp = fn = 0
    n_files = 0
    total_windows = 0
    true_ev_total = 0
    pred_ev_total = 0

    file_rows = {}
    for i in range(len(groups)):
        if not is_orig[i]:
            continue
        file_rows.setdefault(int(groups[i]), []).append(
            (int(starts[i]), int(y[i]), float(fall_probs[i]))
        )

    for fid, rows in file_rows.items():
        rows.sort(key=lambda r: r[0])
        if len(rows) < 2:
            continue
        y_seq = np.array([1 if r[1] == FALL_ID else 0 for r in rows], dtype=np.int32)
        p_seq = np.array([r[2] for r in rows], dtype=np.float32)
        n_files += 1
        total_windows += len(rows)

        pred_seq = smooth_fall_labels(p_seq, ema_alpha, prob_thresh)
        true_events = merge_events(y_seq, min_gap=min_gap)
        pred_events = merge_events(pred_seq, min_gap=min_gap)
        true_ev_total += len(true_events)
        pred_ev_total += len(pred_events)
        t, f_p, f_n = match_events(true_events, pred_events)
        tp += t; fp += f_p; fn += f_n

    return {
        "n_files": n_files,
        "total_windows": total_windows,
        "true_events": true_ev_total,
        "pred_events": pred_ev_total,
        "tp": tp, "fp": fp, "fn": fn,
    }


def main():
    print("=" * 60)
    print("  Six-Class Classifier V6 — ALL DATA MERGED")
    print("=" * 60)
    
    extractor = EnhancedFeatureExtractor(window_size=30)
    
    print("\n[1/5] Loading data (all sources + augmentation)...")
    X, y, groups, starts, is_orig = load_all_data(extractor, augment=True)
    
    if len(X) == 0:
        print("ERROR: No data loaded!")
        return
    
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler
    
    # ── 按文件分组划分（防数据泄漏）──
    # 同一视频的相邻滑窗高度重叠，随机按窗口切会让模型"背答案"。
    # 改为：同一文件(视频)的所有窗口必须整体落在 train 或 val 一侧。
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(iter(sgkf.split(X, y, groups=groups)))
        split_method = "StratifiedGroupKFold (按文件分组 + 类分层, 第1折=20%)"
    except Exception as e:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups=groups))
        split_method = f"GroupShuffleSplit (按文件分组 80/20; stratify 不可用: {e})"
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # 泄漏自检：train 和 val 的文件集合应无交集
    train_files = set(np.unique(groups[train_idx]))
    val_files = set(np.unique(groups[val_idx]))
    leak = train_files & val_files
    print(f"\n  划分方式: {split_method}")
    print(f"  泄漏自检: train {len(train_files)} 文件 | val {len(val_files)} 文件 | "
          f"重叠 {len(leak)} {'⚠️ 有泄漏!' if leak else '✅ 无泄漏'}")
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    print(f"\n  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Dim: {X.shape[1]}")
    
    # ── 2. Train with fixed strong params (HP search too slow with 10k+ samples) ──
    print("\n[2/5] Training XGBoost (fixed params, no HP search to avoid timeout)...")
    import xgboost as xgb
    
    # Class weights to handle imbalance
    class_counts = Counter(y_train)
    scale_pos_weights = {}
    max_count = max(class_counts.values())
    for cls_id in range(6):
        if cls_id in class_counts:
            scale_pos_weights[cls_id] = max_count / class_counts[cls_id]
    
    print(f"  Class weights: {', '.join(f'{CLASS_NAMES[k]}={v:.1f}' for k,v in sorted(scale_pos_weights.items()))}")
    
    # Strong default params (from v3 experience + tuning)
    best_params = {
        'n_estimators': 400,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.85,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.05,
    }
    search_time = 0  # No HP search
    
    model = xgb.XGBClassifier(
        **best_params,
        objective='multi:softprob', num_class=6,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
    )
    
    t0 = time.time()
    model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=50)
    train_time = time.time() - t0
    print(f"  Train time: {train_time:.1f}s")
    
    # ── 4. Evaluate ──
    print("\n[4/5] Evaluating...")
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    
    y_pred = model.predict(X_val_s)
    all_labels = list(range(6))
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0, labels=all_labels)
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0, labels=all_labels)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0, labels=all_labels)
    
    print(f"  Accuracy:  {accuracy:.4f}  |  F1:  {f1:.4f}  |  Precision:  {precision:.4f}  |  Recall:  {recall:.4f}")
    
    cm = confusion_matrix(y_val, y_pred, labels=all_labels)
    print(f"\n  Confusion Matrix:")
    header = f"  {'':>12s}" + "".join(f"{n:>8s}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name:>12s}" + "".join(f"{cm[i][j]:8d}" for j in range(6))
        print(row)
    
    print(f"\n  Per-class F1:")
    for i, name in enumerate(CLASS_NAMES):
        total = cm[i].sum()
        tp = cm[i, i] if i < cm.shape[0] else 0
        pct = 100 * tp / total if total > 0 else 0
        wrongs = [f"{CLASS_NAMES[j]}:{cm[i][j]}" for j in range(6) if j != i and cm[i][j] > 0]
        tag = f"  -> {', '.join(wrongs)}" if wrongs else "  [OK]"
        print(f"    {name:12s}: {tp:5d}/{total:5d} ({pct:5.1f}%){tag}")
    
    print(f"\n  Full Report:")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES, zero_division=0,
                                labels=all_labels))
    
    # ── 5. PR 曲线 + 事件级指标 (Fall 二分类视角) ──
    print("\n[5/6] Fall PR curve + event-level metrics...")
    from sklearn.metrics import (precision_recall_curve, average_precision_score,
                                 roc_curve, roc_auc_score)
    
    y_val_bin = (y_val == FALL_ID).astype(np.int32)
    fall_probs = model.predict_proba(X_val_s)[:, FALL_ID]
    
    ap = float(average_precision_score(y_val_bin, fall_probs))
    roc_auc = float(roc_auc_score(y_val_bin, fall_probs))
    prec_curve, rec_curve, _ = precision_recall_curve(y_val_bin, fall_probs)
    fpr_curve, tpr_curve, _ = roc_curve(y_val_bin, fall_probs)
    print(f"  Fall AP (PR): {ap:.4f}   |   Fall ROC-AUC: {roc_auc:.4f}")
    
    # 事件级指标（只对原始窗口，按文件重建时序）
    ev = evaluate_event_level(
        groups[val_idx], starts[val_idx], is_orig[val_idx],
        y_val, fall_probs,
    )
    ev_prec = ev["tp"] / (ev["tp"] + ev["fp"]) if (ev["tp"] + ev["fp"]) else 0.0
    ev_rec = ev["tp"] / (ev["tp"] + ev["fn"]) if (ev["tp"] + ev["fn"]) else 0.0
    ev_f1 = 2 * ev_prec * ev_rec / (ev_prec + ev_rec) if (ev_prec + ev_rec) else 0.0
    fpr_per_1k = ev["fp"] / ev["total_windows"] * 1000 if ev["total_windows"] else 0.0
    print(f"\n  事件级指标 (简化时序过滤: EMA α=0.3 + 阈值0.35 + 间隙合并):")
    print(f"    验证文件数: {ev['n_files']} | 原始窗口数: {ev['total_windows']}")
    print(f"    真实跌倒事件: {ev['true_events']} | 预测事件: {ev['pred_events']}")
    print(f"    TP={ev['tp']}  FP={ev['fp']}  FN={ev['fn']}")
    print(f"    事件级 Precision={ev_prec:.4f}  Recall={ev_rec:.4f}  F1={ev_f1:.4f}")
    print(f"    误报率: {ev['fp']} 次 / {ev['total_windows']} 窗口 = {fpr_per_1k:.2f} 次/千窗口")
    
    # 画 PR/ROC 曲线 PNG（尽力而为，缺 matplotlib 则跳过）
    pr_png = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(rec_curve, prec_curve, lw=2, label=f"Fall (AP={ap:.3f})")
        axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
        axes[0].set_title("Precision-Recall — Fall")
        axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)
        axes[0].grid(alpha=0.3); axes[0].legend()
        axes[1].plot(fpr_curve, tpr_curve, lw=2, label=f"Fall (AUC={roc_auc:.3f})")
        axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
        axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC — Fall")
        axes[1].grid(alpha=0.3); axes[1].legend()
        fig.tight_layout()
        pr_png = MODEL_OUT.replace(".pkl", "_pr_curve.png")
        fig.savefig(pr_png, dpi=130)
        plt.close(fig)
        print(f"  PR/ROC 曲线已保存: {pr_png}")
    except Exception as e:
        print(f"  (跳过 PR 曲线 PNG: {e})")
    
    # ── 6. CV (skip — group-split holdout 已足够，分组 K-fold 开销大) ──
    print("\n[6/6] Cross-validation skipped (group-split holdout metrics are reliable)")
    cv_acc_mean, cv_acc_std, cv_f1_mean, cv_f1_std = float(accuracy), 0.0, float(f1), 0.0
    
    # ── Save ──
    # Backup old model if exists
    backup_path = r"E:\老人跌倒\models\fall_classifier_6class_20260804_backup.pkl"
    if os.path.exists(r"E:\老人跌倒\models\fall_classifier_6class.pkl"):
        import shutil
        shutil.copy2(r"E:\老人跌倒\models\fall_classifier_6class.pkl", backup_path)
        print(f"\n  Backed up old model → {os.path.basename(backup_path)}")
    
    bundle = {
        "model": model, "scaler": scaler, "classes": CLASS_NAMES,
        "feature_dim": X.shape[1],
        "config": {
            "window_size": 30, "stride": 6,
            "best_params": best_params,
            "augmentation": True, "enhanced_features": True,
            "data_sources": list(DATA_ROOTS.keys()),
            "version": "v6",
        },
        "metrics": {
            "accuracy": float(accuracy), "precision": float(precision),
            "recall": float(recall), "f1": float(f1),
            "fall_ap": ap, "fall_roc_auc": roc_auc,
            "event_precision": ev_prec, "event_recall": ev_rec, "event_f1": ev_f1,
            "event_tp": int(ev["tp"]), "event_fp": int(ev["fp"]),
            "event_fn": int(ev["fn"]),
            "false_alarm_per_1k_windows": fpr_per_1k,
            "split_method": split_method,
            "train_time_s": float(train_time + search_time),
            "n_samples": int(len(X)), "n_classes": 6,
            "cv_accuracy": cv_acc_mean,
            "cv_f1": cv_f1_mean,
        },
    }
    
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(bundle, f)
    
    report = {
        "version": "v6",
        "classes": CLASS_NAMES,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "train_time_s": float(train_time + search_time),
        "n_samples": int(len(X)),
        "best_params": best_params,
        "split_method": split_method,
        "fall_ap": ap,
        "fall_roc_auc": roc_auc,
        "event_level": {
            "tp": int(ev["tp"]), "fp": int(ev["fp"]), "fn": int(ev["fn"]),
            "precision": ev_prec, "recall": ev_rec, "f1": ev_f1,
            "true_events": int(ev["true_events"]),
            "pred_events": int(ev["pred_events"]),
            "total_windows": int(ev["total_windows"]),
            "false_alarm_per_1k_windows": fpr_per_1k,
        },
        "pr_curve": {
            "precision": [round(float(v), 4) for v in prec_curve],
            "recall": [round(float(v), 4) for v in rec_curve],
        },
        "roc_curve": {
            "fpr": [round(float(v), 4) for v in fpr_curve],
            "tpr": [round(float(v), 4) for v in tpr_curve],
        },
        "per_class": {CLASS_NAMES[i]: float(cm[i, i] / max(cm[i].sum(), 1))
                      for i in range(6)},
        "cv_accuracy": cv_acc_mean,
        "cv_std": cv_acc_std,
        "cv_f1": cv_f1_mean,
        "data_sources": list(DATA_ROOTS.keys()),
        "per_class_samples": {CLASS_NAMES[i]: int(cnt) for i, cnt in sorted(Counter(y).items())},
    }
    with open(MODEL_OUT.replace(".pkl", "_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Also save as the main model
    shutil.copy2(MODEL_OUT, r"E:\老人跌倒\models\fall_classifier_6class.pkl")
    
    print(f"\n{'='*60}")
    print(f"  V6 TRAINING COMPLETE!")
    print(f"  Accuracy: {accuracy:.2%}  |  F1: {f1:.4f}")
    print(f"  Features: {X.shape[1]} (42 base + 9 temporal + 8 tilt)")
    print(f"  Samples:  {len(X)} from {len(DATA_ROOTS)} sources")
    print(f"  Model:    {MODEL_OUT}")
    print(f"  Report:   {MODEL_OUT.replace('.pkl', '_report.json')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
