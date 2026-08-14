#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_urfd_sequences.py — 在 70 个 URFD .npz 序列上评估 ML 推理器

验证标准 (序列级):
  - 跌倒序列   → 任意窗口触发跌倒 ≥ 1 次 → 正确
  - ADL 序列   → 任何窗口触发跌倒 ≥ 1 次 → 误报
"""

import argparse, os, sys, json, time
import numpy as np

_proj_root = os.path.dirname(os.path.abspath(__file__))  # 脚本在项目根目录
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "src"))

from src.ml_fall_detector import MLFallDetector

def evaluate_sequences(data_dir: str, model_path: str,
                       window_size: int = 30, stride: int = 5,
                       threshold: float = 0.5):
    """逐序列评估 ML 推理器"""
    detector = MLFallDetector(model_path, window_size=window_size,
                              stride=stride, threshold=threshold)
    print()

    # 收集所有 npz
    files = sorted([f for f in os.listdir(data_dir)
                    if f.endswith(".npz") and f != "metadata.npz"])
    if not files:
        print("[ERROR] no .npz found in", data_dir)
        return

    results = []
    total_windows = 0
    t0 = time.time()

    for fname in files:
        path = os.path.join(data_dir, fname)
        data = np.load(path, allow_pickle=True)
        lm = data["landmarks"]  # (T, 33, 4)
        true_label = int(data.get("label", 0))  # 1=fall, 0=adl
        T = len(lm)

        detector.reset()
        max_prob = 0.0
        fall_triggered = False

        for i in range(T):
            frame_lm = lm[i]  # (33, 4)
            prob, _, is_fall = detector.update(frame_lm)
            max_prob = max(max_prob, prob)
            if is_fall:
                fall_triggered = True

        predicted = 1 if fall_triggered else 0
        status = "OK" if predicted == true_label else "MISS"

        results.append({
            "name": fname.replace(".npz", ""),
            "label": true_label,
            "predicted": predicted,
            "frames": T,
            "max_prob": float(max_prob),
            "status": status,
            "inferences": detector.total_inferences,
            "avg_ms": detector.get_stats()["avg_ms"],
        })

        # 汇总
        total_windows += detector.total_inferences
        label_str = "FALL" if true_label == 1 else "ADL "
        result_str = f"  {fname}  [true={label_str}]  maxP={max_prob:.4f}  {'✅' if status=='OK' else '❌'}"
        print(result_str)

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print("评估摘要")
    print("=" * 60)

    # 统计
    tp = sum(1 for r in results if r["label"] == 1 and r["predicted"] == 1)
    fn = sum(1 for r in results if r["label"] == 1 and r["predicted"] == 0)
    fp = sum(1 for r in results if r["label"] == 0 and r["predicted"] == 1)
    tn = sum(1 for r in results if r["label"] == 0 and r["predicted"] == 0)

    total_fall = tp + fn
    total_adl = tn + fp

    print(f"  总序列: {len(results)} (跌 {total_fall} + 日常 {total_adl})")
    print(f"  总推理窗口: {total_windows}")
    print(f"  耗时: {elapsed:.2f}s")
    print()
    print(f"  跌倒序列: TP={tp}  FN(漏检)={fn}  Recall={tp/max(total_fall,1)*100:.1f}%")
    print(f"  日常序列: TN={tn}  FP(误报)={fp}  Specificity={tn/max(total_adl,1)*100:.1f}%")

    # 漏检详情
    missed = [r for r in results if r["label"] == 1 and r["predicted"] == 0]
    if missed:
        print(f"\n  漏检序列 ({len(missed)}):")
        for r in missed:
            print(f"    {r['name']}: {r['frames']}帧, maxP={r['max_prob']:.4f}")

    # 误报详情
    false_pos = [r for r in results if r["label"] == 0 and r["predicted"] == 1]
    if false_pos:
        print(f"\n  误报序列 ({len(false_pos)}):")
        for r in false_pos:
            print(f"    {r['name']}: {r['frames']}帧, maxP={r['max_prob']:.4f}")

    # 概率分布
    fall_max_probs = [r["max_prob"] for r in results if r["label"] == 1]
    adl_max_probs = [r["max_prob"] for r in results if r["label"] == 0]
    if fall_max_probs:
        print(f"\n  跌倒序列 maxP 分布: min={min(fall_max_probs):.4f} "
              f"median={np.median(fall_max_probs):.4f} "
              f"mean={np.mean(fall_max_probs):.4f} max={max(fall_max_probs):.4f}")
    if adl_max_probs:
        print(f"  日常序列 maxP 分布: min={min(adl_max_probs):.4f} "
              f"median={np.median(adl_max_probs):.4f} "
              f"mean={np.mean(adl_max_probs):.4f} max={max(adl_max_probs):.4f}")

    # 建议阈值
    all_adl_sorted = sorted(adl_max_probs)
    all_fall_sorted = sorted(fall_max_probs)
    if all_adl_sorted and all_fall_sorted:
        print(f"\n  阈值建议:")
        print(f"    ADL 99%分位: {all_adl_sorted[int(len(all_adl_sorted)*0.99)]:.4f}")
        print(f"    Fall 1%分位:  {all_fall_sorted[int(len(all_fall_sorted)*0.01)]:.4f}")
        sep = all_adl_sorted[-1] if all_adl_sorted[-1] < all_fall_sorted[0] else -1
        if sep > 0:
            print(f"    ✅ ADL/Fall maxP 完全分离！安全阈值: {sep:.4f}")
        else:
            print(f"    ⚠️ ADL/Fall maxP 有重叠，需要权衡阈值")
            overlap_adl = [p for p in all_adl_sorted if p > all_fall_sorted[0]]
            overlap_fall = [p for p in all_fall_sorted if p < all_adl_sorted[-1]]
            print(f"    重叠 ADL: {len(overlap_adl)}, 重叠 Fall: {len(overlap_fall)}")
            safe_threshold = all_adl_sorted[-1] + 0.1
            safe_recall = sum(1 for p in all_fall_sorted if p >= safe_threshold) / len(all_fall_sorted)
            print(f"    建议阈值: {safe_threshold:.4f} (Recall {safe_recall*100:.0f}%)")

    print()
    print("=" * 60)

    # 保存报告
    report_path = os.path.join(_proj_root, "models", "eval_urfd_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                       "recall": tp / max(total_fall, 1),
                       "specificity": tn / max(total_adl, 1),
                       "total_sequences": len(results),
                       "total_windows": total_windows,
                       "elapsed_s": round(elapsed, 2)},
            "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  报告: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="URFD 序列级评估 ML 推理器")
    parser.add_argument("--data_dir", "-d", type=str,
                        default="E:/老人跌倒/data/urfd_features/",
                        help="URFD .npz 目录")
    parser.add_argument("--model", "-m", type=str,
                        default="E:/老人跌倒/models/fall_classifier.pkl",
                        help="模型文件路径")
    parser.add_argument("--window", "-w", type=int, default=30)
    parser.add_argument("--stride", "-s", type=int, default=5)
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="跌倒概率阈值 (默认 0.5)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"[ERROR] 模型不存在: {args.model}")
        print("  请先运行: python training/train_fall_classifier.py --data_dir ...")
        return 1

    evaluate_sequences(
        data_dir=args.data_dir,
        model_path=args.model,
        window_size=args.window,
        stride=args.stride,
        threshold=args.threshold,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
