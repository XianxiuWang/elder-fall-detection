"""
护龄 —— 数据录制与标注工具

在摄像头前做动作 + 按键盘打标签，一键录制训练数据。

用法:
    python data_capture.py                     # 录制新数据
    python data_capture.py --output my_data    # 指定输出文件名
    python data_capture.py --camera 1          # 使用外接摄像头

操作说明:
    1-6 键:  切换当前状态标签（walking/sitting/lying/long_sit/abnormal/fall）
    0 键:    标记为"无人/无效帧"（不记录）
    S 键:    开始/暂停录制
    Q 键:    退出程序
    R 键:    重置特征提取器（切换视频/场景后）

画面显示:
    左上角:   当前标签 + 录制状态
    右上角:   已录制帧数
    底部:     状态提示
"""

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from config import (
    DATA_DIR, STATE_HOTKEYS, STATE_LABELS, STATE_NAMES,
    CAMERA_WIDTH, CAMERA_HEIGHT, MODEL_COMPLEXITY,
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
)
from feature_extractor import FeatureExtractor, landmarks_from_mediapipe


# ============================================================
# 界面绘制
# ============================================================
def draw_ui(frame, state_name, is_recording, frame_count, fps, hint=""):
    """在画面上叠加 UI 信息"""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # 半透明顶栏
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # 状态标签（左上）
    state_colors = {
        "walking": (0, 255, 0),
        "sitting": (255, 255, 0),
        "lying": (255, 255, 0),
        "long_sit": (0, 200, 255),
        "abnormal": (0, 140, 255),
        "fall": (0, 0, 255),
        "none": (128, 128, 128),
    }
    color = state_colors.get(state_name, (255, 255, 255))
    label_text = f"State: {state_name}"
    cv2.putText(frame, label_text, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 录制指示灯
    if is_recording:
        rec_color = (0, 0, 255)
        cv2.circle(frame, (15, 55), 8, rec_color, -1)
        cv2.putText(frame, "REC", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec_color, 2)

    # 已录制帧数（右上）
    cv2.putText(frame, f"Frames: {frame_count}", (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 200, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # 底部提示
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    if hint:
        cv2.putText(frame, hint, (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # 快捷键提示（右下）
    shortcuts = "1-6:Label  0:None  S:Record  R:Reset  Q:Quit"
    cv2.putText(frame, shortcuts, (w - 580, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    return frame


def draw_skeleton(frame, landmarks, mp_pose, mp_drawing, mp_drawing_styles):
    """绘制 MediaPipe 骨架"""
    if landmarks is None:
        return frame
    mp_drawing.draw_landmarks(
        frame,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )
    return frame


# ============================================================
# 数据保存
# ============================================================
class DataRecorder:
    """负责将特征向量保存为 CSV"""

    def __init__(self, output_path: str, extractor: FeatureExtractor):
        self.output_path = output_path
        self.extractor = extractor
        self.file = None
        self.writer = None
        self.frame_count = 0
        self._header_written = False

    def start(self):
        self.file = open(self.output_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        self._header_written = False
        self.frame_count = 0

    def record_frame(self, feature_vector, state_name: str):
        if not self.writer:
            return

        # 首行写表头
        if not self._header_written:
            header = list(self.extractor.feature_names()) + ["label", "label_name"]
            self.writer.writerow(header)
            self._header_written = True

        row = list(feature_vector.values) + [STATE_LABELS[state_name], state_name]
        self.writer.writerow(row)
        self.frame_count += 1

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================
# 主循环
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="护龄 - 数据录制工具")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 文件名（默认按时间自动命名）")
    parser.add_argument("--camera", type=int, default=0,
                        help="摄像头索引（默认 0）")
    parser.add_argument("--no-skeleton", action="store_true",
                        help="不显示骨架（性能较低的机器）")
    args = parser.parse_args()

    # 输出文件
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"huling_data_{timestamp}.csv"
    output_path = os.path.join(DATA_DIR, args.output)

    # 初始化
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    extractor = FeatureExtractor(use_motion=False)  # 录制时不计算运动（每帧独立）
    recorder = DataRecorder(output_path, extractor)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f" 无法打开摄像头 (索引 {args.camera})")
        return

    # 状态变量
    current_label = "none"
    is_recording = False
    is_paused = False
    fps = 0.0
    fps_counter = 0
    fps_timer = time.time()
    hint = "按 S 开始录制，1-6 选择状态标签"

    print("=" * 60)
    print("  护龄 - 数据录制与标注工具")
    print("=" * 60)
    print(f"\n  输出文件: {output_path}")
    print(f"\n  操作说明:")
    print(f"    1-6: 选择状态标签")
    for key, name in STATE_HOTKEYS.items():
        if key != ord('0'):
            print(f"      {chr(key)} = {name}")
    print(f"    0 = none (无人)")
    print(f"    S = 开始/停止录制")
    print(f"    R = 重置特征提取器")
    print(f"    Q = 退出")
    print(f"\n  建议录制顺序:")
    print(f"    ① walking  - 在镜头前走动 30-60秒")
    print(f"    ② sitting  - 坐下保持 30秒")
    print(f"    ③ lying    - 躺下（床/沙发）30秒")
    print(f"    ④ fall     - 模拟跌倒（注意安全！）20秒")
    print(f"    ⑤ abnormal - 异常姿态（弯腰、蹲姿）20秒")
    print(f"    ⑥ long_sit - 长时间坐（实际上和 sitting 相似，")
    print(f"                 可以在后期从 sitting 数据中人工构造）")
    print(f"\n  ️ 按 S 开始录制！\n")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as pose:

        while True:
            success, frame = cap.read()
            if not success:
                print(" 读取帧失败")
                break

            # 翻转（镜像效果）
            frame = cv2.flip(frame, 1)

            # MediaPipe 推理
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            has_person = results.pose_landmarks is not None

            # 特征提取 + 录制
            if has_person and is_recording and current_label != "none":
                try:
                    landmarks = landmarks_from_mediapipe(results.pose_landmarks)
                    fv = extractor.extract(landmarks)
                    recorder.record_frame(fv, current_label)
                except Exception as e:
                    hint = f"特征提取出错: {e}"

            # 绘制骨架
            if not args.no_skeleton and has_person:
                frame = draw_skeleton(frame, results.pose_landmarks,
                                      mp_pose, mp_drawing, mp_drawing_styles)

            # 绘制 UI
            total_frames = recorder.frame_count
            frame = draw_ui(frame, current_label, is_recording,
                            total_frames, fps, hint)

            # 如果当前无人，调暗画面提示
            if not has_person and is_recording:
                frame = cv2.addWeighted(frame, 0.6,
                                        np.zeros_like(frame), 0.4, 0)
                hint = "未检测到人体！请站在摄像头前"

            # FPS 计算
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            cv2.imshow("护龄 - 数据录制", frame)

            # 键盘处理
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('s'):
                is_recording = not is_recording
                if is_recording:
                    recorder.start()
                    hint = f" 开始录制! 当前标签: {current_label}"
                else:
                    saved_count = recorder.frame_count
                    recorder.close()
                    print(f"\n 已保存 {saved_count} 帧到 {output_path}")
                    hint = f"录制已停止，共 {saved_count} 帧，按 S 继续录制"

            elif key == ord('r'):
                extractor.reset()
                hint = "特征提取器已重置"

            elif key in STATE_HOTKEYS:
                current_label = STATE_HOTKEYS[key]
                status = " 录制中" if is_recording else "⏸ 暂停"
                hint = f"{status} | 当前标签: {current_label}"

        # 修复 hint 更新 - 记录完后正确显示
        recorded_frames_before_close = recorder.frame_count

    # 清理
    recorder.close()
    cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'=' * 60}")
    print(f"  录制完成!")
    print(f"  文件: {output_path}")
    print(f"  总帧数: {recorded_frames_before_close}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
