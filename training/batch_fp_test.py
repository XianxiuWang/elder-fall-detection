#!/usr/bin/env python3
"""
batch_fp_test.py — FP 压测：在所有非跌倒视频上跑 V6 模型，统计误报率
=====================================================================
用法:
  conda activate fall
  cd /d E:\老人跌倒\training
  set HTTP_PROXY= && set HTTPS_PROXY=
  d:\Anaconda3\envs\fall\python.exe -u batch_fp_test.py

输出:
  fp_test_report.json  — FP 压测汇总报告
  results_*.json       — 每个视频的详细推理结果
"""
import os, sys, json, time
from collections import Counter

sys.path.insert(0, r"E:\老人跌倒\training")
sys.stdout.reconfigure(encoding='utf-8')

# 导入 video_inference 的推理函数
from video_inference import run_inference, CLASS_NAMES, FALL_ID

# ============================================================
# 待测视频清单
# ============================================================
VIDEO_DIR = r"F:\动作数据集"

TEST_VIDEOS = {
    # ── 非跌倒视频（FP 测试目标）──
    "jumping":   os.path.join(VIDEO_DIR, "50个跳跃的动作.mp4"),       # 跳跃 → 不该报 Fall
    "sitting":   os.path.join(VIDEO_DIR, "50个坐下的动作.mp4"),       # 坐下 → 坐下的瞬间易和 Fall 混淆
    "pickup":    os.path.join(VIDEO_DIR, "50 Ways to Pick Up a Dollar50种捡起一美元的方式.mp4"),  # 弯腰捡东西 → 最像 Fall 的非跌倒动作
    # ── 已验证的非跌倒视频（回归确认）──
    "walk":      os.path.join(VIDEO_DIR, "100 Ways to Walk100种走路方式.mp4"),   # 之前 Walk FPR=0%
    "stand":     os.path.join(VIDEO_DIR, "50 Ways to Stand -50种站立方式.mp4"),  # 之前 0 Fall
    # ── 跌倒视频（确认没有漏报）──
    "fall1":     os.path.join(VIDEO_DIR, "50 Ways to Fall50种摔倒方式.mp4"),     # 未单独测过
    "fall2":     os.path.join(VIDEO_DIR, "50个摔倒的动作.mp4"),                 # 未单独测过
}

# ============================================================
# FP 判定规则
# ============================================================
# 非跌倒视频中出现的 Fall 即 FP
NON_FALL_VIDEOS = {"jumping", "sitting", "pickup", "walk", "stand"}
# 跌倒视频中每个 Fall 事件持续 > 2s 才算有效检出
FALL_VIDEOS = {"fall1", "fall2"}

def analyze_results(predictions, video_key, video_name):
    """分析单个视频的推理结果"""
    n_total = len(predictions)
    if n_total == 0:
        return {"error": "No predictions"}

    raw_counts = Counter(r["raw"] for r in predictions)
    filt_counts = Counter(r["filt"] for r in predictions)
    raw_fall = raw_counts.get("Fall", 0)
    filt_fall = filt_counts.get("Fall", 0)

    # 识别 Fall 事件（连续 Fall 帧合并）
    fall_events = []
    for r in predictions:
        if r["filt"] == "Fall":
            t = r["time_s"]
            if not fall_events or t - fall_events[-1]["end"] > 2.0:
                fall_events.append({"start": t, "end": t})
            else:
                fall_events[-1]["end"] = t

    # 取每个事件的最高 fall_prob
    for fe in fall_events:
        fe_probs = [r["fall_prob"] for r in predictions
                    if fe["start"] <= r["time_s"] <= fe["end"]]
        fe["max_prob"] = max(fe_probs) if fe_probs else 0
        fe["duration"] = round(fe["end"] - fe["start"], 1)

    # FP 率计算
    raw_fpr = raw_fall / n_total * 100
    filt_fpr = filt_fall / n_total * 100

    # 判定
    if video_key in NON_FALL_VIDEOS:
        if filt_fall == 0:
            verdict = "✅ PASS — 零误报"
        elif filt_fpr < 3:
            verdict = f"⚠️ WARN — {filt_fpr:.1f}% 误报率"
        else:
            verdict = f"❌ FAIL — {filt_fpr:.1f}% 误报率，需追加难例数据"
    else:
        # 跌倒视频：看是否检测到 Fall
        has_valid = any(fe["duration"] >= 2.0 for fe in fall_events)
        verdict = "✅ PASS — 检测到跌倒" if has_valid else "❌ FAIL — 漏报"

    return {
        "video": video_name,
        "type": "non-fall" if video_key in NON_FALL_VIDEOS else "fall",
        "total_predictions": n_total,
        "raw_falls": raw_fall,
        "filt_falls": filt_fall,
        "raw_fpr_pct": round(raw_fpr, 2),
        "filt_fpr_pct": round(filt_fpr, 2),
        "fall_events": len(fall_events),
        "event_details": fall_events,
        "verdict": verdict,
    }


def main():
    print("=" * 70)
    print("  FP PRESSURE TEST — 跌倒检测误报率压测")
    print(f"  模型: V6 (59维, 98.90%)")
    print(f"  测试视频: {len(TEST_VIDEOS)} 个")
    print("=" * 70)

    t_total_start = time.time()
    results = {}
    summary = []

    for video_key, video_path in TEST_VIDEOS.items():
        print(f"\n{'─'*70}")
        print(f"  [{video_key}] {os.path.basename(video_path)}")

        if not os.path.exists(video_path):
            print(f"  ⚠ SKIP: 文件不存在")
            results[video_key] = {"error": "File not found", "video": os.path.basename(video_path)}
            continue

        t0 = time.time()
        try:
            predictions = run_inference(video_path, save_video=False)
            elapsed = time.time() - t0
            analysis = analyze_results(predictions, video_key, os.path.basename(video_path))
            analysis["elapsed_s"] = round(elapsed, 1)
            analysis["inference_fps"] = round(analysis["total_predictions"] / elapsed, 1)

            print(f"  耗时: {elapsed:.0f}s")
            print(f"  预测窗口: {analysis['total_predictions']}")
            print(f"  Raw Fall: {analysis['raw_falls']} → Filtered Fall: {analysis['filt_falls']}")
            print(f"  FP 率: {analysis['filt_fpr_pct']}%")
            print(f"  Fall 事件: {analysis['fall_events']}")
            for fe in analysis.get("event_details", []):
                print(f"    {fe['start']:6.1f}s - {fe['end']:6.1f}s  "
                      f"({fe['duration']}s, max_prob={fe['max_prob']})")
            print(f"  判定: {analysis['verdict']}")

            results[video_key] = analysis
            summary.append(analysis)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ❌ ERROR ({elapsed:.0f}s): {e}")
            import traceback
            traceback.print_exc()
            results[video_key] = {"error": str(e), "video": os.path.basename(video_path)}

    t_total = time.time() - t_total_start

    # ── 汇总报告 ──
    print(f"\n{'='*70}")
    print(f"  压测完成（总耗时 {t_total:.0f}s）")
    print(f"{'='*70}")

    print(f"\n  {'视频':<12s} {'类型':<8s} {'窗口':>6s} {'Raw':>5s} {'Filt':>5s} {'FPR%':>6s} {'事件':>4s} {'判定'}")
    print(f"  {'-'*65}")
    for s in summary:
        if "total_predictions" in s:
            print(f"  {s['video'][:12]:<12s} {s['type']:<8s} {s['total_predictions']:6d} "
                  f"{s['raw_falls']:5d} {s['filt_falls']:5d} {s['filt_fpr_pct']:6.2f} "
                  f"{s['fall_events']:4d} {s['verdict']}")
        else:
            print(f"  {s.get('video', '?'):<12s} {'ERROR':<8s}")

    # ── 非跌倒视频专项统计 ──
    non_fall_results = [s for s in summary if s.get("type") == "non-fall" and "total_predictions" in s]
    if non_fall_results:
        total_preds = sum(s["total_predictions"] for s in non_fall_results)
        total_filt_falls = sum(s["filt_falls"] for s in non_fall_results)
        overall_fpr = total_filt_falls / total_preds * 100 if total_preds > 0 else 0

        print(f"\n  ── 非跌倒视频汇总 ──")
        print(f"  总预测窗口: {total_preds}")
        print(f"  总误报 Fall: {total_filt_falls}")
        print(f"  综合 FP 率: {overall_fpr:.3f}%")

        if overall_fpr == 0:
            print(f"  ✅ 零误报！模型可以部署。")
        elif overall_fpr < 1:
            print(f"  ⚠️ FP 率 < 1%，可接受但建议监控。")
        elif overall_fpr < 3:
            print(f"  ⚠️ FP 率 {overall_fpr:.1f}%，需要针对性增加难例数据。")
        else:
            print(f"  ❌ FP 率 {overall_fpr:.1f}%，不可部署，必须先降 FP。")

    # ── 保存报告 ──
    report = {
        "model": "V6 (59维, 98.90%)",
        "total_elapsed_s": round(t_total, 1),
        "videos": results,
        "summary": {
            "total_videos": len(summary),
            "non_fall_fpr": overall_fpr if non_fall_results else None,
            "pass_count": sum(1 for s in summary if "PASS" in s.get("verdict", "")),
            "warn_count": sum(1 for s in summary if "WARN" in s.get("verdict", "")),
            "fail_count": sum(1 for s in summary if "FAIL" in s.get("verdict", "")),
        }
    }

    report_path = r"E:\老人跌倒\training\fp_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  完整报告: {os.path.abspath(report_path)}")

    return report


if __name__ == "__main__":
    main()
