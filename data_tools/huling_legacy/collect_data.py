"""
护龄 — 数据采集工具
====================
对着摄像头做动作 → MediaPipe 提取 33 个关键点 → 保存为 .npy 数据集

用法：
    # 采集全部 7 类状态（每类录一段）
    python collect_data.py

    # 只采集特定类别
    python collect_data.py --states 0,5

    # 播放已有视频来提取（不录新的）
    python collect_data.py --video D:/videos/fall_sample.mp4 --label 5

工作流程（每类状态）：
    1. 终端显示"准备录制 → 行走"
    2. 倒计时 3 秒
    3. 录制 5 秒（你在镜头前做对应动作）
    4. 数据保存为 data/state_{id}_{timestamp}.npy
    5. 下一类动作，按 Enter 继续
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import time
from datetime import datetime
from pathlib import Path

import config


# ─── 颜色定义 ───
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)


def draw_ui(frame, state_id, countdown, recording, frame_in_seq):
    """在摄像画面上叠加采集UI"""
    h, w = frame.shape[:2]

    # 顶部半透明条
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 1, overlay, 0.5, 0)

    state_name = config.STATE_NAMES.get(state_id, f"未知({state_id})")
    status_color = RED if recording else GREEN

    cv2.putText(frame, f"状态: {state_name}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    if countdown > 0:
        cv2.putText(frame, f"准备... {countdown}",
                    (w // 2 - 60, h // 2), cv2.FONT_HERSHEY_DUPLEX,
                    1.5, RED, 3)
    elif recording:
        progress_bar = "#" * (frame_in_seq // 2) + "-" * (config.COLLECT_SEQUENCE_LENGTH // 2 - frame_in_seq // 2)
        cv2.putText(frame, f"采集中 [{frame_in_seq}/{config.COLLECT_SEQUENCE_LENGTH}]",
                    (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, RED, 2)

    return frame


def extract_keypoints(results):
    """
    从 MediaPipe Pose 结果中提取关键点坐标
    返回: numpy array shape (33, 3) — (x, y, z, normalized)
    如果未检测到人体，返回全零数组
    """
    if results.pose_landmarks is None:
        return np.zeros((33, 3), dtype=np.float32)

    keypoints = np.zeros((33, 3), dtype=np.float32)
    for i, lm in enumerate(results.pose_landmarks.landmark):
        keypoints[i] = [lm.x, lm.y, lm.z]
    return keypoints


def collect_from_camera(state_id, seq_length=None, fps=None):
    """
    从摄像头采集一个状态的多帧关键点序列

    Args:
        state_id: 状态标签 (0-6)
        seq_length: 序列长度（帧数）
        fps: 采样帧率

    Returns:
        numpy array shape (seq_length, 33, 3) 或 None（失败）
    """
    if seq_length is None:
        seq_length = config.COLLECT_SEQUENCE_LENGTH
    if fps is None:
        fps = config.COLLECT_FPS

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("❌ 无法打开摄像头！")
        return None

    mp_pose = mp.solutions.pose

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=config.MP_MODEL_COMPLEXITY,
        min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MP_MIN_TRACKING_CONFIDENCE,
    ) as pose:

        # ── 倒计时阶段 ──
        for countdown in [3, 2, 1]:
            start_t = time.time()
            while time.time() - start_t < 1.0:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

                frame = draw_ui(frame, state_id, countdown, False, 0)
                cv2.imshow("护龄 数据采集 | 按 Q 退出", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return None

        # ── 采集阶段 ──
        sequences = []
        frame_in_seq = 0
        interval = 1.0 / fps

        while frame_in_seq < seq_length:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 画骨架
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style())

            # 提取并保存关键点
            kp = extract_keypoints(results)
            sequences.append(kp)
            frame_in_seq += 1

            # 显示进度
            frame = draw_ui(frame, state_id, 0, True, frame_in_seq)
            cv2.imshow("护龄 数据采集 | 按 Q 退出", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None

            # 控制帧率
            elapsed = time.time() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)

    cap.release()
    cv2.destroyAllWindows()

    data = np.array(sequences, dtype=np.float32)
    return data


def collect_from_video(video_path, label, seq_length=None, fps=None):
    """
    从已有视频文件中提取关键点序列

    Args:
        video_path: 视频文件路径
        label: 状态标签
    """
    if seq_length is None:
        seq_length = config.COLLECT_SEQUENCE_LENGTH

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")

    mp_pose = mp.solutions.pose
    all_keypoints = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=config.MP_MODEL_COMPLEXITY,
        min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MP_MIN_TRACKING_CONFIDENCE,
    ) as pose:

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            kp = extract_keypoints(results)
            all_keypoints.append(kp)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  处理中... {frame_idx}/{total_frames} 帧")

    cap.release()

    all_keypoints = np.array(all_keypoints, dtype=np.float32)

    # 滑动窗口切分为多个 seq_length 长度的片段
    sequences = []
    step = seq_length // 2  # 50% 重叠
    for start in range(0, len(all_keypoints) - seq_length + 1, step):
        sequences.append(all_keypoints[start:start + seq_length])
        if len(sequences) >= 100:  # 一个视频最多取100条
            break

    if not sequences:
        print(f"⚠️  视频太短（{len(all_keypoints)}帧 < {seq_length}帧），无法切分")
        return None

    print(f"从视频提取了 {len(sequences)} 条序列")
    return np.array(sequences, dtype=np.float32)


def save_sequence(data, label):
    """保存单条序列到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"state_{label}_{timestamp}.npy"
    filepath = config.DATA_DIR / filename
    np.save(filepath, data)
    print(f"  ✅ 已保存: {filepath} ({data.shape})")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="护龄 数据采集工具")
    parser.add_argument("--states", type=str, default=None,
                        help="要采集的状态ID，逗号分隔，如 '0,1,5'。不传则采集全部7类")
    parser.add_argument("--video", type=str, default=None,
                        help="从视频文件提取（不采集新数据）")
    parser.add_argument("--label", type=int, default=0,
                        help="视频对应的标签（仅--video时有效）")
    args = parser.parse_args()

    print("=" * 60)
    print("  护龄 v3 — 数据采集工具")
    print("=" * 60)
    print()

    # ── 从视频提取模式 ──
    if args.video:
        data = collect_from_video(args.video, args.label)
        if data is not None:
            for seq in data:
                save_sequence(seq, args.label)
        return

    # ── 摄像头采集模式 ──
    if args.states:
        state_ids = [int(s.strip()) for s in args.states.split(",")]
    else:
        state_ids = list(range(7))

    print("即将采集以下状态:")
    for sid in state_ids:
        print(f"  [{sid}] {config.STATE_NAMES[sid]}")
    print()
    print("📋 操作指南:")
    print("   · 每类状态录制前有3秒倒计时准备")
    print("   · 倒计时结束后开始录制（5秒左右）")
    print("   · 请在镜头前做出对应动作")
    print("   · 按 Enter 进入下一类，按 Q 随时退出")
    print()

    for i, state_id in enumerate(state_ids):
        print(f"\n{'─' * 50}")
        print(f"[{i+1}/{len(state_ids)}] 准备录制: [{state_id}] {config.STATE_NAMES[state_id]}")

        state_hints = {
            0: "请在摄像头前来回走动、慢走、快走交替",
            1: "请坐在椅子上，保持正常坐姿（可看手机、看书等）",
            2: "请躺在沙发/床上（侧面朝向摄像头效果最好）",
            3: "请长时间保持坐姿不动（约5秒完全静止）",
            4: "请做出弯腰捡东西、蹲下、跪坐等不常见姿势",
            5: "⚠️ 请模拟跌倒（从站立坐到地上/床上，注意安全！）",
            6: "请离开摄像头范围，让画面中无人",
        }
        print(f"  💡 {state_hints.get(state_id, '')}")
        input("  按 Enter 开始...")

        data = collect_from_camera(state_id)
        if data is None:
            print("  ⚠️ 采集被中断，跳过此类")
            continue

        save_sequence(data, state_id)

    print(f"\n{'=' * 60}")
    print("✅ 采集完成！")
    print(f"数据保存目录: {config.DATA_DIR}")
    print(f"下一步: python extract_features.py  # 提取特征")
    print(f"        python train.py             # 训练模型")


if __name__ == "__main__":
    main()
