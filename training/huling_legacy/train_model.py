"""
护龄 —— 模型训练流水线

从录制数据 / 开源数据集 → 特征提取 → 训练 → 评估 → 导出模型。

支持的数据格式:
  1. 自录 CSV（data_capture.py 输出的格式）
  2. 从原始视频 + 标签直接提取（MediaPipe → 特征 → 训练）

用法:
    # 从 CSV 训练
    python train_model.py --csv data/huling_data_20260507.csv

    # 从视频文件夹训练（文件夹按类别组织）
    python train_model.py --video-dir data/videos/ --labels labels.csv

    # 指定模型类型
    python train_model.py --csv data/xxx.csv --model xgboost
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# 尝试导入可选的模型
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.svm import SVC
    HAS_SVM = True
except ImportError:
    HAS_SVM = False

from config import (
    DATA_DIR, MODEL_DIR, STATE_NAMES, STATE_LABELS,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, TEST_SIZE, RANDOM_SEED,
)


# ============================================================
# 数据加载
# ============================================================
def load_csv_data(csv_path: str) -> tuple:
    """
    加载 data_capture.py 输出的 CSV 文件。

    Returns
    -------
    (X, y, feature_names, label_names)
    """
    print(f"\n 加载数据: {csv_path}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)

    # 自动检测 CSV 格式:
    #   格式A: [features..., label, label_name]           (data_capture.py 输出)
    #   格式B: [features..., label, label_name, split]    (process_urfd_images.py 输出)
    has_split_col = (header[-1] == 'split')
    if has_split_col:
        label_col_offset = -3
        feature_names = header[:-3]
    else:
        label_col_offset = -2
        feature_names = header[:-2]

    X_list = []
    y_list = []
    split_list = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            if len(row) < 3:
                continue
            features = [float(v) for v in row[:label_col_offset]]
            label = int(row[label_col_offset])
            X_list.append(features)
            y_list.append(label)
            if has_split_col:
                split_list.append(row[-1])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    print(f"  样本数: {len(X)}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  类别分布:")
    for i, name in enumerate(STATE_NAMES):
        count = (y == i).sum()
        if count > 0:
            print(f"    {name}: {count} 条 ({count/len(y)*100:.1f}%)")

    if has_split_col:
        print(f"  数据划分:")
        from collections import Counter
        for k, v in sorted(Counter(split_list).items()):
            print(f"    {k}: {v} 条 ({v/len(split_list)*100:.1f}%)")

    return X, y, feature_names


def check_class_balance(y: np.ndarray, min_samples: int = 10) -> list:
    """检查类别平衡性，返回样本不足的类别"""
    warnings = []
    for i, name in enumerate(STATE_NAMES):
        count = (y == i).sum()
        if count == 0:
            warnings.append(f" 类别 '{name}' 没有任何样本！")
        elif count < min_samples:
            warnings.append(f" 类别 '{name}' 仅有 {count} 个样本，可能不足")
    return warnings


# ============================================================
# 模型训练
# ============================================================
def create_model(model_type: str):
    """创建指定类型的分类器"""
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError("请先安装 xgboost: pip install xgboost")
        return XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softmax",
            num_class=len(STATE_NAMES),
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    elif model_type == "svm":
        if not HAS_SVM:
            raise ImportError("SVM 应该随 sklearn 安装")
        return SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")


def train_model(X, y, model_type="random_forest", do_scaling=True):
    """
    完整训练流程：划分 → 标准化 → 训练 → 评估。

    Returns
    -------
    (model, scaler, metrics_dict)
    """
    print(f"\n{'=' * 60}")
    print(f"  训练模型: {model_type}")
    print(f"{'=' * 60}")

    # 1. 类别检查
    warnings = check_class_balance(y)
    for w in warnings:
        print(f"\n{w}")

    # 2. 划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    print(f"\n 数据划分:")
    print(f"  训练集: {len(X_train)} 条")
    print(f"  验证集: {len(X_val)} 条")

    # 3. 标准化
    scaler = StandardScaler() if do_scaling else None
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val

    # 4. 训练
    model = create_model(model_type)
    t0 = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - t0

    # 5. 验证集评估
    y_pred = model.predict(X_val_scaled)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

    print(f"\n⏱ 训练用时: {train_time:.2f} 秒")
    print(f"\n 验证集结果:")
    print(f"  准确率 (Accuracy):  {accuracy:.2%}")
    print(f"  精确率 (Precision): {precision:.2%}")
    print(f"  召回率 (Recall):    {recall:.2%}")
    print(f"  F1 分数:            {f1:.2%}")

    # 6. 分类报告
    print(f"\n 各类别详细报告:")
    try:
        present_labels = sorted(set(y_val) | set(y_pred))
        target_names = [STATE_NAMES[i] for i in present_labels]
        report = classification_report(
            y_val, y_pred,
            labels=present_labels,
            target_names=target_names,
            zero_division=0
        )
        print(report)
    except Exception as e:
        print(f"  (报告生成失败: {e})")

    # 7. 混淆矩阵
    cm = confusion_matrix(y_val, y_pred, labels=sorted(set(y_val)))
    print(f"\n 混淆矩阵:")
    labels_used = [STATE_NAMES[i] for i in sorted(set(y_val))]
    header = " " * 10 + "".join(f"{name:>8s}" for name in labels_used)
    print(header)
    for i, name in enumerate(labels_used):
        row = f"{name:10s}" + "".join(f"{cm[i][j]:8d}" for j in range(len(labels_used)))
        print(row)

    # 8. 交叉验证
    print(f"\n 5 折交叉验证...")
    try:
        if scaler:
            X_all_scaled = scaler.fit_transform(X)
        else:
            X_all_scaled = X
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = cross_val_score(model, X_all_scaled, y, cv=cv, scoring="f1_weighted")
        print(f"  CV F1 分数: {cv_scores}")
        print(f"  均值: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    except Exception as e:
        print(f"  (交叉验证跳过: {e})")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "train_time": train_time,
        "model_type": model_type,
        "n_features": X.shape[1],
        "n_samples": len(X),
        "class_distribution": {STATE_NAMES[i]: int((y == i).sum()) for i in range(len(STATE_NAMES))},
    }

    return model, scaler, metrics


# ============================================================
# 模型保存与加载
# ============================================================
def save_model(model, scaler, metrics, name="pose_classifier"):
    """保存完整模型包 (模型 + 标准化器 + 元信息)"""
    model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
    bundle = {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "state_names": STATE_NAMES,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(bundle, model_path)
    file_size = os.path.getsize(model_path) / 1024
    print(f"\n 模型已保存: {model_path}")
    print(f"   文件大小: {file_size:.1f} KB")
    return model_path


def load_model(name="pose_classifier"):
    """加载完整模型包"""
    model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    bundle = joblib.load(model_path)
    return bundle


# ============================================================
# 特征重要性分析（仅随机森林）
# ============================================================
def show_feature_importance(model, feature_names, top_n=20):
    """打印最重要的特征"""
    if not hasattr(model, "feature_importances_"):
        print("\n(当前模型不支持特征重要性查看)")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n Top {top_n} 重要特征:")
    print(f"{'排名':6s} {'特征名':30s} {'重要性':10s}")
    print("-" * 50)
    for i in range(min(top_n, len(indices))):
        idx = indices[i]
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"{i+1:4d}.  {name:30s} {importances[idx]:.4f}")


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="护龄 - 模型训练")
    parser.add_argument("--csv", type=str, default=None,
                        help="训练数据 CSV 路径")
    parser.add_argument("--model", type=str, default="random_forest",
                        choices=["random_forest", "xgboost", "svm"],
                        help="模型类型")
    parser.add_argument("--output", type=str, default="pose_classifier",
                        help="输出模型名称")
    parser.add_argument("--no-scale", action="store_true",
                        help="不进行标准化")
    args = parser.parse_args()

    if args.csv is None:
        # 尝试找到最新数据文件
        csv_files = sorted(Path(DATA_DIR).glob("huling_data_*.csv"))
        if not csv_files:
            print(" 没有找到训练数据文件！")
            print(f"   请先用 data_capture.py 录制数据，或指定 --csv 路径")
            print(f"   数据目录: {DATA_DIR}")
            sys.exit(1)
        args.csv = str(csv_files[-1])
        print(f" 自动选择最新数据文件: {os.path.basename(args.csv)}")

    # 加载数据
    X, y, feature_names = load_csv_data(args.csv)

    # 训练
    model, scaler, metrics = train_model(
        X, y, model_type=args.model, do_scaling=not args.no_scale
    )

    # 特征重要性
    show_feature_importance(model, feature_names)

    # 保存
    save_model(model, scaler, metrics, name=args.output)

    # 输出使用提示
    print(f"\n{'=' * 60}")
    print(f"  后续步骤:")
    print(f"  1. 用 inference.py 实时推理测试")
    print(f"  2. 如果准确率不足：")
    print(f"     - 增加各类别录制数据量")
    print(f"     - 尝试 --model xgboost 换模型")
    print(f"     - 检查混淆矩阵中的误分类模式")
    print(f"  3. 模型文件: models/{args.output}.joblib")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
