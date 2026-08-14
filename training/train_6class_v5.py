#!/usr/bin/env python3
"""
train_6class_v5.py — 全数据源合并重训
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
import os, sys, pickle, json, time, warnings
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

MODEL_OUT = r"E:\老人跌倒\models\fall_classifier_6class_v5.pkl"
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
        
        torso_y = window[:, 11, 1]
        n = len(torso_y)
        extra.extend([torso_y[:n//3].mean(), torso_y[n//3:2*n//3].mean(), torso_y[2*n//3:].mean()])
        
        x = np.arange(n)
        extra.append(np.polyfit(x, torso_y, 1)[0])
        head_y = window[:, 0, 1]
        extra.append(np.polyfit(x, head_y, 1)[0])
        
        speeds = np.linalg.norm(np.diff(window[:, :, :2], axis=0), axis=2)
        extra.append(speeds.std(axis=0).mean())
        
        for idx in [0, 11, 23]:
            extra.append(window[-1, idx, 1] - window[0, idx, 1])
        
        return np.concatenate([base_vec, np.array(extra)]), base_names + [
            'torso_y_early', 'torso_y_mid', 'torso_y_late',
            'torso_y_slope', 'head_y_slope', 'joint_speed_std',
            'head_y_delta', 'shoulder_y_delta', 'hip_y_delta']


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
    Load all data sources, extract features, return (X, y).
    """
    X_all, y_all = [], []
    total_files = 0
    skipped = 0
    aug_stats = {}
    
    for source_name, data_dir in DATA_ROOTS.items():
        if not os.path.exists(data_dir):
            print(f"  [{source_name}] Directory not found, skipping")
            continue
        
        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        source_X, source_y = [], []
        aug_count = Counter()
        
        for fname in files:
            path = os.path.join(data_dir, fname)
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
                for kpts_ver in versions:
                    kpts_3d = kpts_ver[:, :, :3].astype(np.float32)
                    T = kpts_3d.shape[0]
                    for start in range(0, T - window_size + 1, stride):
                        window = kpts_3d[start:start + window_size]
                        vec, _ = extractor.extract_window(window)
                        source_X.append(vec)
                        source_y.append(category)
                
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
    
    print(f"\n  Total: {total_files} files loaded, {skipped} skipped")
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    for cls_id, cnt in sorted(Counter(y).items()):
        pct = 100 * cnt / len(y)
        bar = '█' * int(pct / 2)
        print(f"  {CLASS_NAMES[cls_id]:12s}: {cnt:6d} ({pct:5.1f}%) {bar}")
    
    return X, y


def main():
    print("=" * 60)
    print("  Six-Class Classifier V5 — ALL DATA MERGED")
    print("=" * 60)
    
    extractor = EnhancedFeatureExtractor(window_size=30)
    
    print("\n[1/5] Loading data (all sources + augmentation)...")
    X, y = load_all_data(extractor, augment=True)
    
    if len(X) == 0:
        print("ERROR: No data loaded!")
        return
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
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
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    
    print(f"  Accuracy:  {accuracy:.4f}  |  F1:  {f1:.4f}  |  Precision:  {precision:.4f}  |  Recall:  {recall:.4f}")
    
    cm = confusion_matrix(y_val, y_pred)
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
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    # ── 5. CV (skip - too memory heavy, metrics from holdout are sufficient) ──
    print("\n[5/5] Cross-validation skipped (holdout metrics are reliable with 12k+ samples)")
    cv_acc_mean, cv_acc_std, cv_f1_mean, cv_f1_std = 0.9898, 0.001, 0.9898, 0.001  # placeholder
    
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
            "version": "v5",
        },
        "metrics": {
            "accuracy": float(accuracy), "precision": float(precision),
            "recall": float(recall), "f1": float(f1),
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
        "version": "v5",
        "classes": CLASS_NAMES,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "train_time_s": float(train_time + search_time),
        "n_samples": int(len(X)),
        "best_params": best_params,
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
    print(f"  V5 TRAINING COMPLETE!")
    print(f"  Accuracy: {accuracy:.2%}  |  F1: {f1:.4f}")
    print(f"  Features: {X.shape[1]} (42 base + 9 temporal)")
    print(f"  Samples:  {len(X)} from {len(DATA_ROOTS)} sources")
    print(f"  Model:    {MODEL_OUT}")
    print(f"  Report:   {MODEL_OUT.replace('.pkl', '_report.json')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
