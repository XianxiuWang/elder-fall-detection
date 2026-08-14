#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e2e_lite_baseline.py — 端到端精简验证: 稳定跟踪 + 逐人步行基线校准
目标: 验证"ID 碎片化修复后, 持续人物能在足够帧数内把 PersonalizedBaseline 校准成功
(calibrated:true)"。直接复用真实 GaitMetricExtractor + PersonalBaseline, 用真实 MediaPipe
关键点喂入, 与 e2e 的关键路径一致 (但跳过其他检测器以提速)。

用法(在 E:\\老人跌倒 下):
    python -u src\\e2e_lite_baseline.py -s "F:\\动作数据集\\人类摔倒五花八门.mp4" --frame-step 3 --max-frames 400
"""
import sys, os, time
import numpy as np
import cv2

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from src.stable_tracker import StablePersonTracker
from src.gait_trend import GaitMetricExtractor
from src.personalized_baseline import PersonalBaseline, BaselineConfig

# 用与 e2e 一致的 100 帧步行校准需求 (也可临时调低看是否达标)
WALKING_NEEDED = 100


def feats_from_landmarks(lm, prev_lm):
    """镜像 e2e 的真实特征构造 (feat_vec, 与 e2e_fall_monitor 一致):
    feat[1]=hip 均值, feat[3]=torso 角, 膝角等。真实步长/速度由 extractor 从 landmarks 计算。
    """
    f = np.zeros(96, dtype=np.float32)
    if lm is None or lm.shape[0] < 33:
        return f
    f[1] = lm[23:25, 1].mean()                    # centroid_height (hip 均值 y)
    # torso angle (复制 e2e._compute_torso_angle)
    sh_mid = (lm[11, :2] + lm[12, :2]) / 2
    hi_mid = (lm[23, :2] + lm[24, :2]) / 2
    dy = hi_mid[1] - sh_mid[1]
    dx = hi_mid[0] - sh_mid[0]
    f[3] = float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))
    # 膝角 (23-25-27 与 24-26-28)
    def angle(a, b, c):
        v1 = np.array([a[0]-b[0], a[1]-b[1]]); v2 = np.array([c[0]-b[0], c[1]-b[1]])
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6: return 170.0
        c = np.clip(np.dot(v1, v2)/(n1*n2), -1, 1)
        return float(np.degrees(np.arccos(c)))
    f[72] = angle(lm[23], lm[25], lm[27])
    f[73] = angle(lm[24], lm[26], lm[28])
    f[83] = 0.2      # 对称性
    f[91] = 3.0      # 摆动
    return f


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", "-s", required=True)
    ap.add_argument("--frame-step", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--walking-needed", type=int, default=WALKING_NEEDED)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("[ERROR] 无法打开:", args.source)
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = StablePersonTracker(confidence=0.45)
    extractor = GaitMetricExtractor()
    cfg = BaselineConfig(calibration_min_walking=args.walking_needed)
    baselines = {}      # pid -> PersonalBaseline
    counts = {}         # pid -> {frames, walking}
    prev_lm = {}        # pid -> prev landmarks

    frame_idx = 0
    processed = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and processed >= args.max_frames:
            break
        if frame_idx % args.frame_step != 0:
            frame_idx += 1
            continue
        persons = tracker.update(frame)
        processed += 1

        for p in persons:
            if not p.has_landmarks or p.landmarks is None:
                continue
            pid = p.person_id
            if pid not in baselines:
                baselines[pid] = PersonalBaseline(person_id=f"P{pid}", config=cfg)
                counts[pid] = {"frames": 0, "walking": 0}
            counts[pid]["frames"] += 1
            feats = feats_from_landmarks(p.landmarks, prev_lm.get(pid))
            sample = extractor.extract(feats, landmarks=p.landmarks,
                                       prev_landmarks=prev_lm.get(pid))
            prev_lm[pid] = p.landmarks
            if sample.is_walking:
                counts[pid]["walking"] += 1
            baselines[pid].update(sample)

        if processed % 100 == 0:
            print(f"  [proc {processed}/{total}] {time.time()-t0:.0f}s "
                  f"ids={len(baselines)}", flush=True)
        frame_idx += 1

    cap.release()
    tracker.close()

    print("\n===== 基线校准验证 =====")
    print(f"处理帧: {processed}, 单人ID数: {len(baselines)}")
    cal = sum(1 for b in baselines.values() if b.is_calibrated)
    print(f"已校准(calibrated): {cal}/{len(baselines)}")
    print("\n每人帧数统计 (Pid | 总帧 | 步行帧 | 已校准?):")
    for pid, b in baselines.items():
        c = counts[pid]
        flag = "YES" if b.is_calibrated else "no"
        print(f"  P{pid:>3} | {c['frames']:>5} | {c['walking']:>5} | {flag}")
    return cal, len(baselines)


if __name__ == "__main__":
    main()
