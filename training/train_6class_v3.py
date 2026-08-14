#!/usr/bin/env python3
"""
train_6class_v3.py — 六分类终极打磨版
========================================
改进点:
  1. XGBoost 替代 LightGBM (+0.34%)
  2. 超参随机搜索 (50 iterations)
  3. StandUp 数据增强 (翻转 + 时间翘曲 + 噪声)
  4. 新增时序差分特征 (torso_velocity_profile 等)
"""
import os, sys, pickle, json, time, warnings
from collections import Counter
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_fall_classifier import FeatureExtractor, TrainConfig

DATA_DIR = r"E:\老人跌倒\data\custom_6class"
MODEL_OUT = r"E:\老人跌倒\models\fall_classifier_6class.pkl"
CLASS_NAMES = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"]

import __main__
__main__.TrainConfig = TrainConfig


# ============================================================
# 增强版特征提取器 (新增时序差分特征)
# ============================================================
class EnhancedFeatureExtractor(FeatureExtractor):
    def __init__(self, window_size=30):
        super().__init__(window_size=window_size)
    
    def extract_window(self, window):
        base_vec, base_names = super().extract_window(window)
        extra = []
        
        # torso_y 时间剖面 (前/中/后)
        torso_y = window[:, 11, 1]
        n = len(torso_y)
        extra.extend([torso_y[:n//3].mean(), torso_y[n//3:2*n//3].mean(), torso_y[2*n//3:].mean()])
        
        # torso_y 趋势斜率
        x = np.arange(n)
        extra.append(np.polyfit(x, torso_y, 1)[0])
        
        # head_y 趋势斜率
        head_y = window[:, 0, 1]
        extra.append(np.polyfit(x, head_y, 1)[0])
        
        # 速度剖面标准差
        speeds = np.linalg.norm(np.diff(window[:, :, :2], axis=0), axis=2)
        extra.append(speeds.std(axis=0).mean())
        
        # 关键节点高度变化率
        for idx in [0, 11, 23]:  # head, shoulder, hip
            extra.append(window[-1, idx, 1] - window[0, idx, 1])
        
        extra_names = ['torso_y_early', 'torso_y_mid', 'torso_y_late',
                       'torso_y_slope', 'head_y_slope', 'joint_speed_std',
                       'head_y_delta', 'shoulder_y_delta', 'hip_y_delta']
        
        return np.concatenate([base_vec, np.array(extra)]), base_names + extra_names


# ============================================================
# 数据增强
# ============================================================
def augment_standup(kpts, n_augment=3):
    """StandUp 增强: 翻转 + 时间翘曲 + 噪声"""
    augmented = [kpts.copy()]
    
    # 水平翻转
    flipped = kpts.copy()
    flipped[:, :, 0] = 1.0 - flipped[:, :, 0]
    augmented.append(flipped)
    
    # 时间翘曲
    n_frames = kpts.shape[0]
    if n_frames > 15:
        warped_idx = np.sort(np.random.choice(n_frames, size=n_frames, replace=True))
        augmented.append(kpts[warped_idx].copy())
    
    # 高斯噪声
    noisy = kpts.copy()
    noisy[:, :, :2] += np.random.normal(0, 0.005, noisy[:, :, :2].shape).astype(np.float32)
    augmented.append(noisy)
    
    return augmented[:n_augment + 1]


def load_all_data(data_dir, extractor, window_size=30, stride=5, augment=True):
    X_all, y_all = [], []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    print(f"Found {len(files)} .npz files")
    
    standup_files = 0
    standup_versions = 0
    
    for fname in files:
        path = os.path.join(data_dir, fname)
        try:
            data = np.load(path, allow_pickle=True)
            kpts = data['keypoints']
            category = int(data['category'])
            n_frames = kpts.shape[0]
            if n_frames < window_size:
                continue
            
            # StandUp (category=2 in CLASS_NAMES) augmentation
            versions = [kpts]
            if augment and category == 2:
                versions = augment_standup(kpts, n_augment=3)
                standup_files += 1
                standup_versions += len(versions)
            
            for kpts_ver in versions:
                kpts_3d = kpts_ver[:, :, :3]
                for start in range(0, kpts_ver.shape[0] - window_size + 1, stride):
                    window = kpts_3d[start:start + window_size]
                    vec, _ = extractor.extract_window(window)
                    X_all.append(vec)
                    y_all.append(category)
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")

    if augment and standup_files > 0:
        print(f"  StandUp: {standup_files} files → {standup_versions} versions (augmented)")
    
    X = np.nan_to_num(np.array(X_all, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y_all, dtype=np.int32)
    
    print(f"\nLoaded: {X.shape[0]} samples, {X.shape[1]} features")
    for cls_id, cnt in sorted(Counter(y).items()):
        print(f"  {CLASS_NAMES[cls_id]:12s}: {cnt:6d} ({100*cnt/len(y):5.1f}%)")
    
    return X, y


def main():
    print("=" * 60)
    print("  Six-Class Classifier v3 — POLISHED")
    print("=" * 60)
    
    extractor = EnhancedFeatureExtractor(window_size=30)
    
    print("\n[1/5] Loading data (enhanced features + StandUp augmentation)...")
    X, y = load_all_data(DATA_DIR, extractor, augment=True)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    print(f"\n  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Features: {X.shape[1]}")
    
    # ── 2. 超参搜索 ──
    print("\n[2/5] Hyperparameter search (XGBoost, 50 iters)...")
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    param_dist = {
        'n_estimators': randint(150, 500),
        'max_depth': randint(4, 14),
        'learning_rate': uniform(0.01, 0.1),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
    }
    
    base_xgb = xgb.XGBClassifier(
        objective='multi:softprob', num_class=6,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
    )
    
    t0 = time.time()
    search = RandomizedSearchCV(
        base_xgb, param_dist, n_iter=50, cv=3,
        scoring='accuracy', n_jobs=-1, random_state=42, verbose=1,
    )
    search.fit(X_train_s, y_train)
    search_time = time.time() - t0
    
    print(f"\n  Search time: {search_time:.1f}s")
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV: {search.best_score_:.4f}")
    
    # ── 3. 训练最终模型 ──
    print("\n[3/5] Training final XGBoost...")
    model = xgb.XGBClassifier(
        **search.best_params_,
        objective='multi:softprob', num_class=6,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
    )
    
    t0 = time.time()
    model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    train_time = time.time() - t0
    print(f"  Train time: {train_time:.1f}s")
    
    # ── 4. 评估 ──
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
    header = f"  {'':>10s}" + "".join(f"{n:>8s}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name:>10s}" + "".join(f"{cm[i][j]:8d}" for j in range(min(6, cm.shape[1])))
        print(row)
    
    print(f"\n  Per-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        total = cm[i].sum() if i < cm.shape[0] else 1
        tp = cm[i, i] if i < cm.shape[0] else 0
        pct = 100 * tp / total if total > 0 else 0
        wrongs = [f"{CLASS_NAMES[j]}:{cm[i][j]}" for j in range(6) if j != i and cm[i][j] > 0]
        tag = f"  → {', '.join(wrongs)}" if wrongs else "  OK"
        print(f"    {name:10s}: {tp:4d}/{total:4d} ({pct:5.1f}%){tag}")
    
    print(f"\n  Classification Report:")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    # ── 5. CV ──
    print("[5/5] Cross-validation...")
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_X = np.vstack([X_train_s, X_val_s])
    all_y = np.concatenate([y_train, y_val])
    
    cv_acc = []
    cv_f1 = []
    for metric in ['accuracy', 'f1_weighted']:
        scores = cross_val_score(model, all_X, all_y, cv=cv, scoring=metric, n_jobs=-1)
        if metric == 'accuracy':
            cv_acc = scores
        else:
            cv_f1 = scores
        print(f"  CV {metric}: {scores.mean():.4f} ± {scores.std():.4f} (fold scores: {scores.round(4).tolist()})")
    
    # ── Save ──
    bundle = {
        "model": model, "scaler": scaler, "classes": CLASS_NAMES,
        "feature_dim": X.shape[1],
        "config": {"window_size": 30, "stride": 5, "best_params": search.best_params_,
                   "augmentation": True, "enhanced_features": True},
        "metrics": {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
                    "train_time_s": train_time + search_time,
                    "n_samples": len(X), "n_classes": 6},
    }
    
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(bundle, f)
    
    report = {
        "classes": CLASS_NAMES, "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "train_time_s": train_time + search_time, "n_samples": int(len(X)),
        "best_params": {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
                        for k, v in search.best_params_.items()},
        "per_class": {CLASS_NAMES[i]: float(cm[i, i] / max(cm[i].sum(), 1))
                      for i in range(min(6, cm.shape[0]))},
        "cv_accuracy": float(cv_acc.mean()), "cv_f1": float(cv_f1.mean()),
    }
    with open(MODEL_OUT.replace(".pkl", "_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"  POLISH COMPLETE!")
    print(f"  Accuracy: {accuracy:.2%}  |  F1: {f1:.4f}  |  CV: {cv_acc.mean():.4f}")
    print(f"  Features: {X.shape[1]} (42 base + 9 enhanced)")
    print(f"  Samples:  {len(X)} (+{len(X)-2908} via StandUp augmentation)")
    print(f"  Model:    {MODEL_OUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
