#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracker_bench.py — 仅跟踪(跳过 MediaPipe)的快速 ID 碎片化验证脚本
用途: 在完整视频上快速统计"ID 碎片化"程度, 不跑逐人 MediaPipe(那是瓶颈)。
这样能在几分钟内跑完整个视频, 得到总 ID 数与持续 ID 数。
"""
import sys, time, os
import numpy as np
import cv2

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # E:\老人跌倒
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)
from src.stable_tracker import StablePersonTracker


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", "-s", required=True)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--frame-step", type=int, default=3,
                    help="隔几帧处理一帧(越大越快)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--min-active-frames", type=int, default=10)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("[ERROR] 无法打开:", args.source)
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_w = int(cap.get(3)); cap_h = int(cap.get(4))
    print(f"[bench] 视频 {cap_w}x{cap_h}, {total} 帧")

    # 用 monkey-patch 跳过 MediaPipe: 在 update 里不调用 _pose_on_box
    tracker = StablePersonTracker(confidence=args.conf)
    tracker._pose_on_box = lambda *a, **k: (None, False, 0.0)

    t0 = time.time()
    frame_idx = 0
    processed = 0
    last_report = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and processed >= args.max_frames:
            break
        if frame_idx % args.frame_step != 0:
            frame_idx += 1
            continue
        tracker.update(frame)
        processed += 1
        if processed - last_report >= 100:
            last_report = processed
            st = tracker.get_status()
            el = time.time() - t0
            print(f"  [proc {processed}/{total}] active={st['active_count']} "
                  f"tracked={st['tracked_count']} reuse={st['reuse_count']} "
                  f"hist_ids={st['history_ids']} {el:.0f}s", flush=True)
        frame_idx += 1
    cap.release()
    tracker.close()

    counts = tracker.get_id_counts()
    total_ids = len(counts)
    active_ids = sum(1 for c in counts.values() if c >= args.min_active_frames)
    print("\n===== ID 碎片化统计 (跳过 MediaPipe) =====")
    print(f"总生成 ID 数: {total_ids}")
    print(f"活跃 ID 数(出现>={args.min_active_frames}帧): {active_ids}")
    print(f"处理帧数: {processed}, 用时: {time.time()-t0:.0f}s")
    if counts:
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
        print("Top 持续 ID (Pid: 帧数):")
        for pid, cnt in top:
            print(f"  P{pid}: {cnt}")
    return total_ids, active_ids


if __name__ == "__main__":
    main()
