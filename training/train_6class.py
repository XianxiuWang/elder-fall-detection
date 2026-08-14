#!/usr/bin/env python3
"""
train_6class.py — 六分类模型训练入口
=====================================
1. 调用 prepare_6class 准备数据
2. 修改训练参数训练六分类模型
3. 输出模型到 models/fall_classifier_6class.pkl

用法:
    python -m training.train_6class
    python -m training.train_6class --no-prepare  # 跳过数据准备
"""

import os
import sys
import argparse
import subprocess

_PROJ_ROOT = r"E:\老人跌倒"
sys.path.insert(0, _PROJ_ROOT)

DATA_DIR = os.path.join(_PROJ_ROOT, "data", "custom_6class")
MODEL_OUT = os.path.join(_PROJ_ROOT, "models", "fall_classifier_6class.pkl")

# 六分类的 CLASS_NAMES（需要覆盖训练脚本的默认值）
CLASS_NAMES_6 = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"]


def run_prepare():
    print("[1/3] Preparing 6-class data...")
    prepare_script = os.path.join(_PROJ_ROOT, "training", "prepare_6class.py")
    result = subprocess.run(
        [sys.executable, prepare_script, "--min-frames", "300", "--noise-count", "150"],
        cwd=_PROJ_ROOT, capture_output=False
    )
    if result.returncode != 0:
        print("[ERROR] Data preparation failed")
        return False
    print("  Data ready in", DATA_DIR)
    return True


def run_training():
    print("\n[2/3] Training 6-class model...")
    train_script = os.path.join(_PROJ_ROOT, "training", "train_fall_classifier.py")

    # Monkey-patch CLASS_NAMES in the training module to include "Standing"
    import training.train_fall_classifier as tfc
    original_names = list(tfc.CLASS_NAMES) if hasattr(tfc, 'CLASS_NAMES') else None

    # Write a temporary training wrapper
    wrapper = os.path.join(_PROJ_ROOT, "training", "_train_6class_wrapper.py")
    with open(wrapper, 'w', encoding='utf-8') as f:
        f.write('''\
"""Auto-generated 6-class training wrapper."""
import sys, os
sys.path.insert(0, r"E:\\\\老人跌倒")
from training.train_fall_classifier import *

# Override CLASS_NAMES for 6-class
# Note: This must match the FeatureExtractor that loads data
DATA_DIR = r"E:\\老人跌倒\\data\\custom_6class"
MODEL_OUT = r"E:\\老人跌倒\\models\\fall_classifier_6class.pkl"

def main():
    import argparse
    from training.train_fall_classifier import (
        TrainConfig, FeatureExtractor, load_all_npz,
        train_model, evaluate_model, CLASS_NAMES as _CN
    )

    # Override class names
    _CN.clear()
    _CN.extend(["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"])
    global CLASS_NAMES
    CLASS_NAMES = list(_CN)

    print(f"Training 6-class model with classes: {CLASS_NAMES}")
    print(f"Data dir: {DATA_DIR}")

    config = TrainConfig(
        model_type="lgb",
        window_size=30,
        window_stride=5,
        test_ratio=0.2,
        cv_folds=5,
    )

    # Load data
    extractor = FeatureExtractor(window_size=config.window_size)
    X, y, filenames = load_all_npz(DATA_DIR, extractor, stride=config.window_stride)

    if X is None or len(X) == 0:
        print("[ERROR] No training data loaded")
        return

    print(f"Loaded {len(X)} samples, {X.shape[1]} features, {len(set(y))} classes")
    from collections import Counter
    for cls_id, cnt in sorted(Counter(y).items()):
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        print(f"  {cls_name}: {cnt} samples")

    # Train
    model, scaler, metrics = train_model(X, y, [], config)

    # Save
    import pickle, numpy as np
    bundle = {
        "model": model,
        "scaler": scaler,
        "classes": CLASS_NAMES,
        "feature_dim": X.shape[1],
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\\nModel saved to {MODEL_OUT}")

    # Report
    print(f"\\n{'='*60}")
    print("Training Report")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics.get('accuracy', 'N/A')}")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Features: {X.shape[1]}")
    print(f"Model type: {type(model).__name__}")

if __name__ == "__main__":
    main()
''')

    result = subprocess.run(
        [sys.executable, wrapper],
        cwd=_PROJ_ROOT, capture_output=False
    )
    os.remove(wrapper)
    return result.returncode == 0


def update_detector():
    print("\n[3/3] Updating detector for 6-class...")
    detector_path = os.path.join(_PROJ_ROOT, "src", "ml_6class_detector.py")

    # Copy 5class → 6class with updated CLASS_NAMES
    src = os.path.join(_PROJ_ROOT, "src", "ml_5class_detector.py")
    with open(src, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update class names
    content = content.replace(
        'CLASS_NAMES = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp"]',
        'CLASS_NAMES = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"]'
    )
    content = content.replace(
        'CLASS_LABELS_CN = {\n    0: "摔倒",\n    1: "坐下",\n    2: "站起",\n    3: "走路",\n    4: "睡醒",\n}',
        'CLASS_LABELS_CN = {\n    0: "摔倒",\n    1: "坐下",\n    2: "站起",\n    3: "走路",\n    4: "睡醒",\n    5: "站立",\n}'
    )
    content = content.replace(
        "ML5ClassDetector",
        "ML6ClassDetector"
    )
    content = content.replace(
        "5ClassDetector",
        "6ClassDetector"
    )
    content = content.replace(
        "class ML6ClassDetector:",
        'class ML6ClassDetector:\n    """基于 LightGBM 的六分类行为识别推理器（含 Standing/Idle）"""'
    )
    content = content.replace(
        '"""基于 LightGBM 的五分类行为识别推理器"""',
        '"""基于 LightGBM 的六分类行为识别推理器"""'
    )
    content = content.replace(
        '五分类行为识别推理模块',
        '六分类行为识别推理模块'
    )
    content = content.replace(
        '五分类模型',
        '六分类模型'
    )
    content = content.replace(
        'fall_classifier_5class.pkl',
        'fall_classifier_6class.pkl'
    )

    with open(detector_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  Created {detector_path}")
    print("  To use, update e2e_fall_monitor.py:")
    print("    from .ml_6class_detector import ML6ClassDetector")
    print("    self.ml_detector = ML6ClassDetector(model_path, ...)")
    return True


def main():
    parser = argparse.ArgumentParser(description="6-class model training pipeline")
    parser.add_argument("--no-prepare", action="store_true", help="Skip data preparation")
    args = parser.parse_args()

    if not args.no_prepare:
        if not run_prepare():
            sys.exit(1)
    else:
        if not os.path.exists(DATA_DIR):
            print(f"[ERROR] Data dir not found: {DATA_DIR}")
            print("Run without --no-prepare first to generate data")
            sys.exit(1)

    if not run_training():
        print("[WARNING] Training may have issues, check output above")
        # Don't exit - the training wrapper might have worked partially

    update_detector()

    print(f"\n{'='*60}")
    print("6-CLASS MODEL PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {MODEL_OUT}")
    print(f"Detector: src/ml_6class_detector.py")
    print(f"\nNext: Update e2e_fall_monitor.py to use ML6ClassDetector")


if __name__ == "__main__":
    main()
