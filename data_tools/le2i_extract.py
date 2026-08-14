"""
Le2i Fall Detection Dataset — 下载 + 关键点提取 + 格式对齐
================================================================
用法:
  conda activate fall
  python data_tools/le2i_extract.py --data_dir E:/老人跌倒/data/le2i_raw --out_dir E:/老人跌倒/data/urfd_features

数据集来源: http://le2i.cnrs.fr/Fall-detection-dataset
需手动下载后放到 --data_dir 目录下。

目录结构预期:
  le2i_raw/
    Home_01/
      Video/
        fall-01.avi  (或 .mp4)
        adl-01.avi
        ...
    Home_02/
      ...
    Office_01/
      ...
    Coffee_room_01/
      ...

输出:
  urfd_features/le2i_home01_fall01.npz  (每个视频一个 .npz)
  格式与 URFD 完全一致: shape=(T, 33, 4), keypoints_3d, keypoints_scores
"""
import argparse, os, sys, glob, time
import numpy as np
import cv2
import mediapipe as mp


def extract_keypoints_from_video(video_path: str, skip_frames: int = 1):
    """从视频逐帧提取 MediaPipe 33 点关键点 (skip_frames=1 即每帧都取)"""
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        cap = cv2.VideoCapture(video_path)
        all_landmarks = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % skip_frames != 0:
                frame_idx += 1
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                lm_vis = np.zeros((33, 4), dtype=np.float32)
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    lm_vis[i, 0] = lm.x
                    lm_vis[i, 1] = lm.y
                    lm_vis[i, 2] = lm.z
                    lm_vis[i, 3] = lm.visibility  # MediaPipe 内置可见性
                all_landmarks.append(lm_vis)
            frame_idx += 1
        cap.release()

    if len(all_landmarks) == 0:
        return None, 0

    landmarks = np.stack(all_landmarks, axis=0)  # (T, 33, 4)
    return landmarks, len(all_landmarks)


def extract_category(subpath: str) -> int:
    """根据目录名/文件名判断类别: 0=ADL, 1=FALL"""
    lower = subpath.lower()
    if 'fall' in lower:
        return 1
    return 0


def parse_video_name(name: str):
    """解析 Le2i 命名: fall-01-cam0.avi, adl-02-cam1.avi 等"""
    base = os.path.splitext(name)[0]
    parts = base.split('-')
    category = parts[0] if len(parts) > 0 else base
    seq_idx = parts[1] if len(parts) > 1 else '00'
    cam = parts[2] if len(parts) > 2 else ''
    return category, seq_idx, cam


def main():
    parser = argparse.ArgumentParser(description="Le2i → URFD 格式提取器")
    parser.add_argument("--data_dir", default=r"E:\老人跌倒\data\le2i_raw",
                        help="Le2i 原始视频目录")
    parser.add_argument("--out_dir", default=r"E:\老人跌倒\data\urfd_features",
                        help="输出 .npz 目录 (与 URFD 混合)")
    parser.add_argument("--fps", type=int, default=15,
                        help="提取帧率 (默认 15, 即 skip 每2帧)")
    parser.add_argument("--dry_run", action="store_true",
                        help="只扫描不提取")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 扫描所有视频
    video_exts = ('.avi', '.mp4', '.mov', '.mkv', '.webm')
    videos = []
    for root, dirs, files in os.walk(args.data_dir):
        for f in files:
            if f.lower().endswith(video_exts):
                videos.append((root, f))

    if not videos:
        print(f"[ERROR] 未找到视频文件! 目录: {args.data_dir}")
        print("  请确保 Le2i 数据集已下载并解压到此目录。")
        print("  预期结构: Home_01/Video/fall-01.avi 等")
        sys.exit(1)

    print(f"找到 {len(videos)} 个视频文件")

    # 统计
    fall_count = 0
    adl_count = 0
    skip_frames = max(1, 30 // args.fps)  # 假设原视频 30fps

    for root, fname in sorted(videos):
        full_path = os.path.join(root, fname)
        category, seq_idx, cam = parse_video_name(fname)

        # 每个视频只取 cam0 (或第一个摄像机), 避免 8 个视角重复
        # Le2i 一般每个场景单摄像头, 但保留逻辑防万一
        if cam and cam not in ('cam0', '0'):
            print(f"  跳过 {fname} (多视角, 非 cam0)")
            continue

        cat_label = 1 if 'fall' in category.lower() else 0
        cat_name = "FALL" if cat_label else "ADL"

        # 输出文件名: le2i_{场景}_{类别}_{序号}.npz
        scene = os.path.basename(os.path.dirname(root))  # 例如 Home_01
        out_name = f"le2i_{scene}_{category}_{seq_idx}.npz"

        if args.dry_run:
            print(f"  [DRY] {full_path:60s} → {out_name} ({cat_name})")
            continue

        print(f"  提取 {fname:50s} ({cat_name}) ...", end=" ", flush=True)
        t0 = time.time()
        landmarks, n_frames = extract_keypoints_from_video(full_path, skip_frames)
        elapsed = time.time() - t0

        if landmarks is None or n_frames == 0:
            print(f"SKIP (未检测到人体, {elapsed:.1f}s)")
            continue

        # 保存为 URFD 兼容格式
        out_path = os.path.join(args.out_dir, out_name)
        np.savez(
            out_path,
            keypoints=landmarks,  # (T, 33, 4) — URFD 统一字段名
            category=cat_label,
            source="le2i",
            scene=scene,
            orig_name=fname,
        )

        if cat_label:
            fall_count += 1
        else:
            adl_count += 1

        print(f"OK ({n_frames}frames, {elapsed:.1f}s)")

    # 报告
    total = fall_count + adl_count
    print(f"\n{'='*60}")
    print(f"  提取完成: {total} 个序列")
    print(f"    跌倒 (FALL): {fall_count}")
    print(f"    日常 (ADL):  {adl_count}")
    print(f"  输出目录: {args.out_dir}")
    if total == 0:
        print(f"  ⚠️  无视频成功提取！请检查 --data_dir")
    else:
        print(f"  ✅ 可直接用于 train_fall_classifier.py 训练")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
