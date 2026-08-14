#!/usr/bin/env python3
"""
train_6class_v7.py — 全数据源 + 分段StandUp/WakeUp + GroupKFold 5折
====================================================================
相比 v5 的改动:
  1. 新增数据源 custom_standup_wakeup (102 个独立动作段)
     StandUp=52 段, WakeUp=50 段 (从 50 Ways 视频分段提取)
  2. 去重: custom_6class/custom_5class 中 category 2/4 的旧文件全部跳过
     (旧 8 个文件已被新 102 段覆盖)
  3. 划分方式: train_test_split(按窗口,有泄漏) → GroupKFold 5折(按文件分组)
     - 每折验证集都是"没见过的文件", 无数据泄漏
     - 每折都能评估到所有 6 类 (解决 StandUp/WakeUp 验证集为 0 的问题)

数据源:
  - subject_features/  (Fall/SitDown/Walking)
  - custom_6class/     (去重后, Fall/SitDown/Walking/Standing)
  - custom_5class/     (去重后, Fall/SitDown/Walking/Standing...)
  - custom_standup_wakeup/ (StandUp/WakeUp 分段, 新增)
  - urfd_features/     (Fall)
  - le2i_features/     (Fall)
  - upfall_features/   (Fall + Standing)
"""
import os, sys, pickle, json, time, warnings
from collections import Counter
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_fall_classifier import FeatureExtractor

DATA_ROOTS = {
    'subject_features':       r"E:\老人跌倒\data\subject_features",
    'custom_6class':          r"E:\老人跌倒\data\custom_6class",
    'custom_5class':          r"E:\老人跌倒\data\custom_5class",
    'custom_standup_wakeup':  r"E:\老人跌倒\data\custom_standup_wakeup",
    'urfd_features':          r"E:\老人跌倒\data\urfd_features",
    'le2i_features':          r"E:\老人跌倒\data\le2i_features",
    'upfall_features':        r"E:\老人跌倒\data\upfall_features",
}

# 这些数据源里 category 2/4 的旧文件被新数据源替代, 全部跳过
DEDUP_SOURCES = ('custom_6class', 'custom_5class')

MODEL_OUT = r"E:\老人跌倒\models\fall_classifier_6class_v7.pkl"
CLASS_NAMES = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"]
N_FOLDS = 5


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


def augment_heavy(kpts, n_augment=5):
    versions = [kpts.copy()]
    T, J, _ = kpts.shape
    flipped = kpts.copy()
    flipped[:, :, 0] = 1.0 - flipped[:, :, 0]
    versions.append(flipped)
    if T > 15:
        warped_idx = np.sort(np.random.choice(T, size=T, replace=True))
        versions.append(kpts[warped_idx].copy())
    noisy = kpts.copy()
    noisy[:, :, :2] += np.random.normal(0, 0.005, noisy[:, :, :2].shape).astype(np.float32)
    versions.append(noisy)
    scaled = kpts.copy()
    scale = 1.0 + np.random.normal(0, 0.03)
    scaled[:, :, :2] *= scale
    versions.append(scaled)
    if n_augment >= 5:
        mir_noisy = flipped.copy()
        mir_noisy[:, :, :2] += np.random.normal(0, 0.005, mir_noisy[:, :, :2].shape).astype(np.float32)
        versions.append(mir_noisy)
    return versions[:n_augment + 1]


def augment_light(kpts, n_augment=2):
    versions = [kpts.copy()]
    flipped = kpts.copy()
    flipped[:, :, 0] = 1.0 - flipped[:, :, 0]
    versions.append(flipped)
    if n_augment >= 2:
        noisy = kpts.copy()
        noisy[:, :, :2] += np.random.normal(0, 0.003, noisy[:, :, :2].shape).astype(np.float32)
        versions.append(noisy)
    return versions[:n_augment + 1]


def load_all_data_grouped(extractor, window_size=30, stride=6, augment=True):
    """
    加载所有数据源, 返回 (X, y, groups)。
    groups[i] = 样本 i 所属文件的唯一 id (同一文件的所有窗口+增强共享同一个 group)。
    """
    X_all, y_all, groups_all = [], [], []
    total_files = 0
    skipped = 0
    skipped_dedup = 0
    group_counter = 0

    for source_name, data_dir in DATA_ROOTS.items():
        if not os.path.exists(data_dir):
            print(f"  [{source_name}] Directory not found, skipping")
            continue

        files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        source_n = 0
        aug_count = Counter()

        for fname in files:
            path = os.path.join(data_dir, fname)
            try:
                data = np.load(path, allow_pickle=True)

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

                category = None
                if 'category' in data:
                    cat_val = data['category']
                    if hasattr(cat_val, 'item') and hasattr(cat_val, 'ndim') and cat_val.ndim == 0:
                        category = int(cat_val.item())

                if source_name == 'urfd_features':
                    if 'label' in data:
                        label = int(data['label'].item())
                        if label == 0:
                            continue
                        else:
                            category = 0

                if source_name == 'le2i_features' and 'frame_fall_map' in data:
                    ffm = data['frame_fall_map']
                    if np.all(ffm == 0):
                        continue
                    category = 0

                if category is None or category not in range(6):
                    skipped += 1
                    continue

                # 去重: 旧数据源的 StandUp/WakeUp 由新数据源替代
                if source_name in DEDUP_SOURCES and category in (2, 4):
                    skipped_dedup += 1
                    continue

                versions = [kpts]
                if augment:
                    if category in (2, 4):
                        versions = augment_heavy(kpts, n_augment=5)
                        aug_count['heavy'] += 1
                    elif category == 1:
                        versions = augment_light(kpts, n_augment=2)
                        aug_count['light'] += 1

                n_before = len(X_all)
                for kpts_ver in versions:
                    kpts_3d = kpts_ver[:, :, :3].astype(np.float32)
                    T = kpts_3d.shape[0]
                    for start in range(0, T - window_size + 1, stride):
                        window = kpts_3d[start:start + window_size]
                        vec, _ = extractor.extract_window(window)
                        X_all.append(vec)
                        y_all.append(category)
                        groups_all.append(group_counter)
                source_n += 1
                group_counter += 1
                total_files += 1
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"  [SKIP] {source_name}/{fname}: {e}")

        if source_n:
            print(f"  [{source_name}] {source_n:3d} files → {len(X_all)} cumulative samples | "
                  f"aug={dict(aug_count) if aug_count else 'none'}")

    X = np.nan_to_num(np.array(X_all, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y_all, dtype=np.int32)
    groups = np.array(groups_all, dtype=np.int64)

    print(f"\n  Total: {total_files} files, {skipped} skipped, {skipped_dedup} dedup-skipped (old StandUp/WakeUp)")
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(set(groups))} groups")
    for cls_id, cnt in sorted(Counter(y).items()):
        pct = 100 * cnt / len(y)
        print(f"    {CLASS_NAMES[cls_id]:12s}: {cnt:6d} ({pct:5.1f}%)")
    return X, y, groups


def train_fold(X_train, y_train, X_val, y_val, fold):
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)

    class_counts = Counter(y_train)
    max_count = max(class_counts.values())
    scale_pos_weights = {c: max_count / class_counts[c] for c in class_counts}

    best_params = {
        'n_estimators': 400,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.85,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.05,
    }
    model = xgb.XGBClassifier(
        **best_params,
        objective='multi:softprob', num_class=6,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
    )
    model.fit(X_tr, y_train, eval_set=[(X_va, y_val)], verbose=False)
    return model, scaler


def evaluate(model, scaler, X_val, y_val):
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    X_va = scaler.transform(X_val)
    y_pred = model.predict(X_va)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=list(range(6)))
    recall_per = {CLASS_NAMES[i]: (cm[i, i] / max(cm[i].sum(), 1)) for i in range(6)}
    return acc, f1, cm, recall_per


def main():
    print("=" * 60)
    print("  Six-Class Classifier V7 — GroupKFold 5-fold + segmented StandUp/WakeUp")
    print("=" * 60)

    extractor = EnhancedFeatureExtractor(window_size=30)
    print("\n[1/4] Loading data (grouped by file)...")
    X, y, groups = load_all_data_grouped(extractor, augment=True)

    if len(X) == 0:
        print("ERROR: No data loaded!")
        return

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=N_FOLDS)

    print(f"\n[2/4] GroupKFold {N_FOLDS}-fold (grouped by file, no leakage)...")
    fold_accs, fold_f1s, fold_recalls = [], [], []
    fold_cms = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        val_cnt = Counter(y_va)
        val_str = ", ".join(f"{CLASS_NAMES[c]}:{val_cnt.get(c, 0)}" for c in range(6))
        print(f"\n  --- Fold {fold+1}/{N_FOLDS} ---")
        print(f"    train={len(y_tr)}, val={len(y_va)} | val classes: {val_str}")

        t0 = time.time()
        model, scaler = train_fold(X_tr, y_tr, X_va, y_va, fold)
        acc, f1, cm, recall_per = evaluate(model, scaler, X_va, y_va)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_recalls.append(recall_per)
        fold_cms.append(cm)
        print(f"    acc={acc:.4f}  f1={f1:.4f}  ({time.time()-t0:.0f}s)")
        for c in range(6):
            tot = cm[c].sum()
            ok = cm[c, c]
            print(f"      {CLASS_NAMES[c]:12s}: {ok:5d}/{tot:5d} ({100*ok/max(tot,1):5.1f}%)")

    # 汇总
    print(f"\n[3/4] Cross-validation summary ({N_FOLDS}-fold)")
    acc_mean, acc_std = np.mean(fold_accs), np.std(fold_accs)
    f1_mean, f1_std = np.mean(fold_f1s), np.std(fold_f1s)
    print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  F1:       {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"\n  Per-class recall (mean over folds):")
    for c in range(6):
        recalls = [r[CLASS_NAMES[c]] for r in fold_recalls]
        print(f"    {CLASS_NAMES[c]:12s}: {np.mean(recalls)*100:5.1f}% ± {np.std(recalls)*100:.1f}%")

    # 聚合混淆矩阵
    cm_total = np.sum(fold_cms, axis=0)
    print(f"\n  Aggregated confusion matrix (sum over folds):")
    header = f"  {'':>12s}" + "".join(f"{n:>8s}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name:>12s}" + "".join(f"{cm_total[i][j]:8d}" for j in range(6))
        print(row)

    # 最终模型: 全量数据训练
    print(f"\n[4/4] Training final model on FULL data...")
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X)
    class_counts = Counter(y)
    max_count = max(class_counts.values())
    scale_pos_weights = {c: max_count / class_counts[c] for c in class_counts}
    best_params = {
        'n_estimators': 400, 'max_depth': 8, 'learning_rate': 0.05,
        'subsample': 0.85, 'colsample_bytree': 0.8,
        'min_child_weight': 3, 'gamma': 0.05,
    }
    final_model = xgb.XGBClassifier(
        **best_params, objective='multi:softprob', num_class=6,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
    )
    t0 = time.time()
    final_model.fit(X_full, y)
    print(f"  Full-data train time: {time.time()-t0:.1f}s")

    # 保存
    bundle = {
        "model": final_model, "scaler": scaler, "classes": CLASS_NAMES,
        "feature_dim": X.shape[1],
        "config": {
            "window_size": 30, "stride": 6, "best_params": best_params,
            "augmentation": True, "enhanced_features": True,
            "data_sources": list(DATA_ROOTS.keys()),
            "dedup_sources": list(DEDUP_SOURCES),
            "n_folds": N_FOLDS, "version": "v7",
        },
        "metrics": {
            "cv_accuracy_mean": float(acc_mean), "cv_accuracy_std": float(acc_std),
            "cv_f1_mean": float(f1_mean), "cv_f1_std": float(f1_std),
            "n_samples": int(len(X)), "n_classes": 6,
        },
    }
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(bundle, f)

    report = {
        "version": "v7",
        "classes": CLASS_NAMES,
        "cv_accuracy_mean": float(acc_mean),
        "cv_accuracy_std": float(acc_std),
        "cv_f1_mean": float(f1_mean),
        "cv_f1_std": float(f1_std),
        "n_samples": int(len(X)),
        "n_groups": int(len(set(groups))),
        "best_params": best_params,
        "per_class_recall_mean": {CLASS_NAMES[i]: float(np.mean([r[CLASS_NAMES[i]] for r in fold_recalls]))
                                  for i in range(6)},
        "per_class_recall_std": {CLASS_NAMES[i]: float(np.std([r[CLASS_NAMES[i]] for r in fold_recalls]))
                                 for i in range(6)},
        "fold_accuracies": [float(a) for a in fold_accs],
        "fold_f1s": [float(a) for a in fold_f1s],
        "per_class_samples": {CLASS_NAMES[i]: int(cnt) for i, cnt in sorted(Counter(y).items())},
        "data_sources": list(DATA_ROOTS.keys()),
    }
    with open(MODEL_OUT.replace(".pkl", "_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  V7 TRAINING COMPLETE!")
    print(f"  CV Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  CV F1:       {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  Samples: {len(X)} | Groups: {len(set(groups))}")
    print(f"  Model:  {MODEL_OUT}")
    print(f"  Report: {MODEL_OUT.replace('.pkl', '_report.json')}")
    print(f"  (主模型 fall_classifier_6class.pkl 未改动, 确认后再替换)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
