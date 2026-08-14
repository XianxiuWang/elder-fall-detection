#!/usr/bin/env python3
"""
sweep_threshold_v8.py — 阈值扫描: 召回率 vs 误报率 取舍曲线
====================================================================
复用 v8 的加载器 + 事件级评估, GroupKFold 5折收集 OOF(袋外) Fall 概率,
然后在全量 OOF 上扫描 (ema_alpha × prob_thresh) 网格,
输出:
  1. 文本表 (每个组合的 TP/FP/FN/precision/recall/f1/FPR-per-1k)
  2. CSV 文件
  3. 曲线图 (recall vs FPR + precision-recall), 英文标签避免字体问题
"""
import os, sys, time, csv, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_6class_v8 import (
    EnhancedFeatureExtractor, load_all_data_grouped,
    evaluate_event_level, CLASS_NAMES, N_FOLDS,
)

OUT_CSV = r"E:\老人跌倒\models\threshold_sweep_v8.csv"
OUT_PNG = r"E:\老人跌倒\models\threshold_sweep_v8.png"

BEST_PARAMS = {
    'n_estimators': 400, 'max_depth': 8, 'learning_rate': 0.05,
    'subsample': 0.85, 'colsample_bytree': 0.8,
    'min_child_weight': 3, 'gamma': 0.05,
}


def main():
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold

    print("=" * 60)
    print("  Threshold Sweep (v8 protocol) — OOF Fall probs, 5-fold")
    print("=" * 60)

    extractor = EnhancedFeatureExtractor(window_size=30)
    print("\n[1/3] Loading data...")
    X, y, groups, starts, is_orig = load_all_data_grouped(extractor, augment=True)

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_probs = np.full(len(y), np.nan, dtype=np.float64)

    print(f"\n[2/3] GroupKFold {N_FOLDS}-fold → collect OOF fall probs...")
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups)):
        t0 = time.time()
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_va = scaler.transform(X[va_idx])
        model = xgb.XGBClassifier(
            **BEST_PARAMS, objective='multi:softprob', num_class=6,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0,
        )
        model.fit(X_tr, y[tr_idx])
        oof_probs[va_idx] = model.predict_proba(X_va)[:, 0]  # Fall 概率
        print(f"  fold {fold+1}/{N_FOLDS} done ({time.time()-t0:.0f}s)")

    assert not np.isnan(oof_probs).any(), "OOF probs incomplete!"

    # 扫描网格
    print(f"\n[3/3] Sweep (ema_alpha × prob_thresh)...")
    alphas = [0.15, 0.2, 0.3, 0.4, 0.5]
    thresholds = np.round(np.arange(0.10, 0.76, 0.05), 2)

    results = []
    for alpha in alphas:
        for thresh in thresholds:
            ev = evaluate_event_level(
                groups, starts, is_orig, y, oof_probs,
                ema_alpha=alpha, prob_thresh=float(thresh))
            tp, fp, fn = ev['tp'], ev['fp'], ev['fn']
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            fpr = fp / max(ev['total_windows'], 1) * 1000
            results.append(dict(alpha=alpha, thresh=thresh,
                                tp=tp, fp=fp, fn=fn,
                                precision=prec, recall=rec, f1=f1,
                                fpr_per_1k=fpr,
                                total_windows=ev['total_windows'],
                                true_events=ev['true_events']))

    # 文本表 (按 alpha 分组)
    print(f"\n{'='*100}")
    for alpha in alphas:
        print(f"\n  ema_alpha={alpha}")
        print(f"  {'thresh':>7s} {'TP':>5s} {'FP':>5s} {'FN':>4s} "
              f"{'precision':>9s} {'recall':>7s} {'F1':>6s} {'FPR/1k':>7s}")
        for r in results:
            if r['alpha'] != alpha:
                continue
            print(f"  {r['thresh']:7.2f} {r['tp']:5d} {r['fp']:5d} {r['fn']:4d} "
                  f"{r['precision']:9.4f} {r['recall']:7.4f} {r['f1']:6.4f} {r['fpr_per_1k']:7.3f}")

    # 写 CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ema_alpha", "prob_thresh", "tp", "fp", "fn",
            "precision", "recall", "f1", "fpr_per_1k",
            "total_windows", "true_events"])
        w.writeheader()
        for r in results:
            w.writerow({
                "ema_alpha": r["alpha"],
                "prob_thresh": r["thresh"],
                "tp": r["tp"], "fp": r["fp"], "fn": r["fn"],
                "precision": round(r["precision"], 6),
                "recall": round(r["recall"], 6),
                "f1": round(r["f1"], 6),
                "fpr_per_1k": round(r["fpr_per_1k"], 6),
                "total_windows": r["total_windows"],
                "true_events": r["true_events"],
            })
    print(f"\n  CSV → {OUT_CSV}")

    # 曲线图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        markers = ["o", "s", "^", "D", "v"]
        for i, alpha in enumerate(alphas):
            sub = [r for r in results if r["alpha"] == alpha]
            sub.sort(key=lambda r: r["thresh"])
            xs = [r["recall"] for r in sub]
            ys = [r["fpr_per_1k"] for r in sub]
            axes[0].plot(xs, ys, marker=markers[i % len(markers)],
                         label=f"ema_alpha={alpha}")
        axes[0].set_xlabel("Event recall")
        axes[0].set_ylabel("False alarms per 1000 windows")
        axes[0].set_title("Recall vs False-alarm rate")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].invert_xaxis()  # 召回高在左, 符合习惯

        for i, alpha in enumerate(alphas):
            sub = [r for r in results if r["alpha"] == alpha]
            sub.sort(key=lambda r: r["recall"])
            axes[1].plot([r["recall"] for r in sub],
                         [r["precision"] for r in sub],
                         marker=markers[i % len(markers)], label=f"ema_alpha={alpha}")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-Recall")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=130)
        print(f"  PNG → {OUT_PNG}")
    except Exception as e:
        print(f"  [plot skipped] {e}")

    print(f"\n{'='*100}")
    print("  SWEEP COMPLETE")


if __name__ == "__main__":
    main()
