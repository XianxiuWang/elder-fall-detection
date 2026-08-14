#!/usr/bin/env python3
"""
train_6class_v2.py — 六分类模型训练（直接版本）
================================================
从 custom_6class 目录加载所有 .npz，用 LightGBM 训练六分类模型。

用法:
    python train_6class_v2.py
"""

import os
import sys
import pickle
import json
import time
import warnings
from collections import Counter
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_fall_classifier import FeatureExtractor, TrainConfig

DATA_DIR = r"E:\老人跌倒\data\custom_6class"
MODEL_OUT = r"E:\老人跌倒\models\fall_classifier_6class.pkl"
CLASS_NAMES = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"]

# 清除旧的 TrainConfig 引用
import __main__
__main__.TrainConfig = TrainConfig


def load_all_data(data_dir, extractor, window_size=30, stride=5):
    """加载所有 .npz 文件，提取特征和标签"""
    X_all, y_all, sources = [], [], []
    
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    print(f"Found {len(files)} .npz files")
    
    for fname in files:
        path = os.path.join(data_dir, fname)
        try:
            data = np.load(path, allow_pickle=True)
            kpts = data['keypoints']  # (N, 33, 4)
            category = int(data['category'])  # 类别 ID 0-5
            source = str(data.get('source', 'unknown'))

            n_frames = kpts.shape[0]
            if n_frames < window_size:
                continue

            # 取 xyz 三维
            kpts_3d = kpts[:, :, :3]

            # 滑动窗口提取特征
            for start in range(0, n_frames - window_size + 1, stride):
                window = kpts_3d[start:start + window_size]
                vec, _ = extractor.extract_window(window)
                X_all.append(vec)
                y_all.append(category)

        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    
    # 处理 NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\nLoaded: {X.shape[0]} samples, {X.shape[1]} features")
    for cls_id, cnt in sorted(Counter(y).items()):
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        print(f"  {cls_name:12s}: {cnt:6d} samples ({cnt/len(y)*100:.1f}%)")
    
    return X, y


def main():
    print("=" * 60)
    print("  Six-Class Fall Classifier Training")
    print("=" * 60)
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Data: {DATA_DIR}")
    print(f"  Output: {MODEL_OUT}")
    
    # ── 1. 加载数据 ──
    config = TrainConfig(window_size=30, window_stride=5)
    extractor = FeatureExtractor(window_size=30)
    
    print("\n[1/4] Loading data...")
    X, y = load_all_data(DATA_DIR, extractor, window_size=30, stride=5)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    # ── 2. 训练 LightGBM ──
    print("\n[2/4] Training LightGBM 6-class classifier...")
    import lightgbm as lgb
    
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        num_leaves=63,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight='balanced',
        objective='multiclass',
        num_class=6,
        random_state=42,
        verbose=-1,
    )
    
    t0 = time.time()
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_val_s, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")
    
    # ── 3. 评估 ──
    print("\n[3/4] Evaluating...")
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    
    y_prob = model.predict_proba(X_val_s)
    y_pred = model.predict(X_val_s)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    
    # 每类指标
    print("\n  Per-class metrics:")
    cm = confusion_matrix(y_val, y_pred)
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i] if i < cm.shape[0] else 0
        total = cm[i].sum() if i < cm.shape[0] else 1
        acc = tp / total if total > 0 else 0
        print(f"    {name:12s}: {tp:5d}/{total:5d} = {acc:.2%}")
    
    print("\n  Classification Report:")
    # 处理验证集可能不包含所有类别的情况
    present_labels = sorted(set(y_val) | set(y_pred))
    present_names = [CLASS_NAMES[i] for i in present_labels]
    print(classification_report(y_val, y_pred, labels=present_labels,
                                target_names=present_names, zero_division=0))
    
    # 交叉验证
    print("\n[4/4] Cross-validation...")
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for metric in ['accuracy', 'f1_weighted']:
        s = cross_val_score(model, X_train_s, y_train, cv=cv,
                            scoring=metric, n_jobs=1)
        print(f"  CV {metric}: {s.mean():.4f} ± {s.std():.4f}")
    
    # ── 保存模型 ──
    bundle = {
        "model": model,
        "scaler": scaler,
        "classes": CLASS_NAMES,
        "feature_dim": X.shape[1],
        "config": {"window_size": 30, "stride": 5, "n_estimators": 300},
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "train_time_s": train_time,
            "n_samples": len(X),
            "n_classes": 6,
        },
    }
    
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(bundle, f)
    
    report_path = MODEL_OUT.replace(".pkl", "_report.json")
    report = {
        "classes": CLASS_NAMES,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "train_time_s": train_time,
        "n_samples": int(len(X)),
        "per_class": {
            CLASS_NAMES[i]: float(cm[i, i] / max(cm[i].sum(), 1))
            for i in range(min(len(CLASS_NAMES), cm.shape[0]))
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE!")
    print(f"  Accuracy: {accuracy:.2%}  |  F1: {f1:.4f}")
    print(f"  Model: {MODEL_OUT}")
    print(f"  Report: {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
