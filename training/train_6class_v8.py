#!/usr/bin/env python3
"""
train_6class_v8.py — 全数据源 + 分段StandUp/WakeUp/SitDown + GroupKFold 5折
====================================================================
相比 v7 的改动:
  1. 新增数据源 custom_sitdown (50 个独立坐下动作段, 从 50个坐下的动作.mp4 提取)
     - 解决 SitDown 召回率低 (v7: 54.9% ± 24.5%) 的根因: 源文件仅 23 个
  2. 保留 v7 的 custom_standup_wakeup + 去重 + GroupKFold 划分

数据源:
  - subject_features/  (Fall/SitDown/Walking)
  - custom_6class/     (去重后, Fall/SitDown/Walking/Standing)
  - custom_5class/     (去重后, Fall/SitDown/Walking/Standing...)
  - custom_standup_wakeup/ (StandUp/WakeUp 分段)
  - custom_sitdown/    (SitDown 分段, 新增)
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
    'custom_sitdown':         r"E:\老人跌倒\data\custom_sitdown",
    'urfd_features':          r"E:\老人跌倒\data\urfd_features",
    'le2i_features':          r"E:\老人跌倒\data\le2i_features",
    'upfall_features':        r"E:\老人跌倒\data\upfall_features",
}

# 这些数据源里 category 2/4 的旧文件被新数据源替代, 全部跳过
DEDUP_SOURCES = ('custom_6class', 'custom_5class')

MODEL_OUT = r"E:\老人跌倒\models\fall_classifier_6class_v8.pkl"
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
    加载所有数据源, 返回 (X, y, groups, starts, is_orig)。
    groups[i] = 样本 i 所属文件的唯一 id (同一文件的所有窗口+增强共享同一个 group)。
    starts[i] = 样本 i 的窗口起始帧索引 (事件级时序重建用)。
    is_orig[i] = 是否为原始(非增强)窗口。
    """
    X_all, y_all, groups_all = [], [], []
    starts_all, is_orig_all = [], []
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
                for vi, kpts_ver in enumerate(versions):
                    kpts_3d = kpts_ver[:, :, :3].astype(np.float32)
                    T = kpts_3d.shape[0]
                    for start in range(0, T - window_size + 1, stride):
                        window = kpts_3d[start:start + window_size]
                        vec, _ = extractor.extract_window(window)
                        X_all.append(vec)
                        y_all.append(category)
                        groups_all.append(group_counter)
                        starts_all.append(start)
                        is_orig_all.append(vi == 0)
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
    starts = np.array(starts_all, dtype=np.int64)
    is_orig = np.array(is_orig_all, dtype=bool)

    print(f"\n  Total: {total_files} files, {skipped} skipped, {skipped_dedup} dedup-skipped (old StandUp/WakeUp)")
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(set(groups))} groups")
    for cls_id, cnt in sorted(Counter(y).items()):
        pct = 100 * cnt / len(y)
        print(f"    {CLASS_NAMES[cls_id]:12s}: {cnt:6d} ({pct:5.1f}%)")
    return X, y, groups, starts, is_orig


# ============================================================
# 事件级评估工具 (Fall 二分类视角, 对齐部署端 TemporalFilter)
# ============================================================
FALL_ID = 0  # CLASS_NAMES[0] = "Fall"


def smooth_fall_labels(fall_probs, ema_alpha=0.3, prob_thresh=0.35):
    """简化时序过滤：EMA 平滑 + 概率阈值。返回 0/1 序列。
    (不含 fall_inference.py 里的自由落体判别器, 因为此处无原始关键点)"""
    ema = None
    out = []
    for p in fall_probs:
        ema = p if ema is None else ema_alpha * p + (1 - ema_alpha) * ema
        out.append(1 if ema >= prob_thresh else 0)
    return np.array(out, dtype=np.int32)


def merge_events(binary_seq, min_gap=1, min_len=1):
    """把 0/1 序列中的连续 1 段合并为事件 [start, end) 开区间, 允许 min_gap 间隙。"""
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
    """贪心匹配真/预测事件, 按重叠窗口数降序。返回 (tp, fp, fn)。"""
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
    """事件级评估：只对原始窗口, 按文件重建时序 → 简化过滤 → 事件合并匹配。"""
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
    print("  Six-Class Classifier V8 — GroupKFold 5-fold + segmented StandUp/WakeUp/SitDown")
    print("=" * 60)

    extractor = EnhancedFeatureExtractor(window_size=30)
    print("\n[1/4] Loading data (grouped by file)...")
    X, y, groups, starts, is_orig = load_all_data_grouped(extractor, augment=True)

    if len(X) == 0:
        print("ERROR: No data loaded!")
        return

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=N_FOLDS)

    print(f"\n[2/4] GroupKFold {N_FOLDS}-fold (grouped by file, no leakage)...")
    fold_accs, fold_f1s, fold_recalls = [], [], []
    fold_cms = []
    fold_events, fold_aps, fold_aucs = [], [], []

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

        # 事件级评估 (Fall 二分类视角, 仅原始窗口) + AP/ROC-AUC
        from sklearn.metrics import average_precision_score, roc_auc_score
        X_va_s = scaler.transform(X_va)
        fall_probs = model.predict_proba(X_va_s)[:, FALL_ID]
        y_va_bin = (y_va == FALL_ID).astype(np.int32)
        ev = evaluate_event_level(
            groups[va_idx], starts[va_idx], is_orig[va_idx], y_va, fall_probs)
        fold_events.append(ev)
        ap = auc = float('nan')
        if len(np.unique(y_va_bin)) == 2:
            ap = average_precision_score(y_va_bin, fall_probs)
            auc = roc_auc_score(y_va_bin, fall_probs)
        fold_aps.append(ap)
        fold_aucs.append(auc)
        print(f"    event-level (Fall): TP={ev['tp']} FP={ev['fp']} FN={ev['fn']} "
              f"| true_events={ev['true_events']} pred_events={ev['pred_events']}")
        print(f"    Fall AP={ap:.4f}  ROC-AUC={auc:.4f}")

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

    # 事件级汇总 (跨折累加 TP/FP/FN)
    ev_tp = sum(e['tp'] for e in fold_events)
    ev_fp = sum(e['fp'] for e in fold_events)
    ev_fn = sum(e['fn'] for e in fold_events)
    ev_true = sum(e['true_events'] for e in fold_events)
    ev_pred = sum(e['pred_events'] for e in fold_events)
    ev_windows = sum(e['total_windows'] for e in fold_events)
    ev_precision = ev_tp / max(ev_tp + ev_fp, 1)
    ev_recall = ev_tp / max(ev_tp + ev_fn, 1)
    ev_f1 = 2 * ev_precision * ev_recall / max(ev_precision + ev_recall, 1e-9)
    ev_fpr_per_1k = ev_fp / max(ev_windows, 1) * 1000
    print(f"\n  Event-level (Fall, aggregated over folds):")
    print(f"    TP={ev_tp} FP={ev_fp} FN={ev_fn} | true_events={ev_true} pred_events={ev_pred}")
    print(f"    precision={ev_precision:.4f} recall={ev_recall:.4f} f1={ev_f1:.4f}")
    print(f"    false_alarm_per_1k_windows={ev_fpr_per_1k:.4f}")
    print(f"  Fall AP={np.nanmean(fold_aps):.4f} ± {np.nanstd(fold_aps):.4f} | "
          f"ROC-AUC={np.nanmean(fold_aucs):.4f} ± {np.nanstd(fold_aucs):.4f}")

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
            "n_folds": N_FOLDS, "version": "v8",
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
        "version": "v8",
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
        "fall_ap_mean": float(np.nanmean(fold_aps)),
        "fall_ap_std": float(np.nanstd(fold_aps)),
        "fall_roc_auc_mean": float(np.nanmean(fold_aucs)),
        "fall_roc_auc_std": float(np.nanstd(fold_aucs)),
        "event_level": {
            "tp": int(ev_tp), "fp": int(ev_fp), "fn": int(ev_fn),
            "true_events": int(ev_true), "pred_events": int(ev_pred),
            "total_windows": int(ev_windows),
            "precision": round(ev_precision, 4),
            "recall": round(ev_recall, 4),
            "f1": round(ev_f1, 4),
            "false_alarm_per_1k_windows": round(ev_fpr_per_1k, 4),
        },
        "per_class_samples": {CLASS_NAMES[i]: int(cnt) for i, cnt in sorted(Counter(y).items())},
        "data_sources": list(DATA_ROOTS.keys()),
    }
    with open(MODEL_OUT.replace(".pkl", "_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  V8 TRAINING COMPLETE!")
    print(f"  CV Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  CV F1:       {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  Samples: {len(X)} | Groups: {len(set(groups))}")
    print(f"  Model:  {MODEL_OUT}")
    print(f"  Report: {MODEL_OUT.replace('.pkl', '_report.json')}")
    print(f"  (主模型 fall_classifier_6class.pkl 未改动, 确认后再替换)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
