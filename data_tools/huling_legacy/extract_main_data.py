"""
从 E:\main_data 图片数据集提取 MediaPipe 特征，生成训练用 CSV。
优化版：共用 Pose 实例，实时进度打印。
"""

import os, re, sys, time, argparse
from collections import defaultdict

import cv2, mediapipe as mp, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, STATE_LABELS
from feature_extractor import FeatureExtractor, landmarks_from_mediapipe

MAIN_DATA_DIR = r"E:\main_data"


def group_frames(directory):
    """按视频前缀分组帧文件"""
    groups = defaultdict(list)
    pat = re.compile(r'^(.+?)(\d+)\.(png|jpg|jpeg)$', re.I)
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            m = pat.match(fname)
            prefix = m.group(1).rstrip('-').rstrip('_') if m else fname
            groups[prefix].append(fname)
    return dict(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=None)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--data-dir", default=MAIN_DATA_DIR)
    args = ap.parse_args()

    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(DATA_DIR, f"main_data_features_{ts}.csv")
    elif not os.path.isabs(args.output):
        args.output = os.path.join(DATA_DIR, args.output)

    # ---- 收集任务 ----
    tasks = []
    for split in ["train", "valid", "test"]:
        sd = os.path.join(args.data_dir, split)
        if not os.path.isdir(sd):
            continue
        for cls_name in os.listdir(sd):
            cd = os.path.join(sd, cls_name)
            if os.path.isdir(cd):
                label = "fall" if cls_name.lower() == "fall" else "sitting"
                tasks.append((cd, label, split))

    print(f"\n{'='*60}")
    print(f"Extracting from {args.data_dir}")
    print(f"Tasks: {len(tasks)}  |  Stride: {args.stride}  |  MaxFrames: {args.max_frames}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n", flush=True)

    # ---- 共用 MediaPipe & Extractor ----
    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=1,
        smooth_landmarks=True, enable_segmentation=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    extractor = FeatureExtractor(use_motion=True, smooth_window=0)

    all_rows = []
    total_frames, total_detected = 0, 0
    t0 = time.time()

    for task_idx, (img_dir, label_name, split) in enumerate(tasks):
        label_int = STATE_LABELS.get(label_name, 1)
        groups = group_frames(img_dir)
        task_frames, task_detected = 0, 0

        print(f"[{task_idx+1}/{len(tasks)}] {split}/{label_name}  ({len(groups)} videos)", flush=True)

        for vid_name, fnames in sorted(groups.items()):
            extractor.reset()
            vid_records = 0
            for fi, fname in enumerate(fnames[:args.max_frames]):
                if fi % args.stride != 0:
                    continue
                frame = cv2.imread(os.path.join(img_dir, fname))
                if frame is None:
                    continue
                task_frames += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_res = pose.process(rgb)
                if mp_res.pose_landmarks is None:
                    extractor.reset()
                    continue

                task_detected += 1
                lms = landmarks_from_mediapipe(mp_res.pose_landmarks)
                fv = extractor.extract_with_motion(lms)

                row = {"label": label_int, "label_name": label_name, "split": split}
                for j, v in enumerate(fv.values):
                    row[f"f{j}"] = float(v)
                all_rows.append(row)
                vid_records += 1

            if vid_records > 0:
                print(f"  {vid_name}: {vid_records} records", flush=True)

        print(f"  => {task_detected} detected / {task_frames} frames\n", flush=True)
        total_frames += task_frames
        total_detected += task_detected

    pose.close()

    # ---- 保存 ----
    if not all_rows:
        print("ERROR: No features extracted!", flush=True)
        return

    import pandas as pd
    df = pd.DataFrame(all_rows)
    # Move label, label_name, split to end
    cols = [c for c in df.columns if c not in ("label", "label_name", "split")]
    df = df[cols + ["label", "label_name", "split"]]
    df.to_csv(args.output, index=False, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE!  {elapsed:.0f}s  |  {total_detected}/{total_frames} frames with pose  |  {len(all_rows)} records")
    print(f"Output: {args.output}")
    for ln, cnt in df["label_name"].value_counts().items():
        print(f"  {ln}: {cnt}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
