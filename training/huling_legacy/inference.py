"""\nHuLing - Real-time Inference & Visualization\n\nLoad a trained model, run real-time inference on camera, display:\n  - Human skeleton + keypoints\n  - Current state + confidence\n  - 6-class probability bar chart\n  - Fall alert (visual + text flashing)\n  - State history timeline\n\nUsage:\n    python inference.py                          # Use default model\n    python inference.py --model my_model         # Specify model\n    python inference.py --camera 1               # External camera\n    python inference.py --no-display             # Headless mode (print only)\n    python inference.py --save-video output.mp4  # Save inference video\n"""

import argparse
import collections
import os
import sys
import time
from datetime import datetime
from typing import Optional

import cv2
import joblib
import mediapipe as mp
import numpy as np

# Windows 终端按键检测（在 OpenCV 窗口失焦时也能响应 R/Q）
if os.name == 'nt':
    try:
        import msvcrt
        HAS_MSVCRT = True
    except ImportError:
        HAS_MSVCRT = False
else:
    HAS_MSVCRT = False

from config import (
    DATA_DIR, MODEL_DIR, STATE_NAMES, STATE_LABELS, STATE_LABELS_REVERSE,
    CAMERA_WIDTH, CAMERA_HEIGHT, MODEL_COMPLEXITY,
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
)
from feature_extractor import FeatureExtractor, landmarks_from_mediapipe


# ============================================================
# 模型加载
# ============================================================
def load_bundle(model_name="pose_classifier"):
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please run train_model.py first to train the model"
        )
    return joblib.load(model_path)


# ============================================================
# 状态颜色映射
# ============================================================
STATE_COLORS = {
    "walking":   (0, 255, 0),     # 绿
    "sitting":   (255, 255, 0),   # 黄
    "lying":     (200, 200, 0),   # 暗黄
    "long_sit":  (0, 200, 255),   # 橙
    "abnormal":  (0, 140, 255),   # 橙红
    "fall":      (0, 0, 255),     # 红
    "unknown":   (128, 128, 128), # 灰
}

ALERT_LEVEL_COLORS = {
    0: (0, 255, 0),     # 正常 - 绿
    1: (0, 255, 255),   # 低 - 黄
    2: (0, 165, 255),   # 中 - 橙
    3: (0, 0, 255),     # 高 - 红
}

STATE_ALERT_LEVEL = {
    "walking": 0, "sitting": 0, "lying": 0,
    "long_sit": 1, "abnormal": 2, "fall": 3,
    "unknown": 0,
}


# ============================================================
# 推理引擎
# ============================================================
class InferenceEngine:
    """
    实时推理引擎：MediaPipe → 特征提取 → 模型分类 → 输出结果。
    """

    def __init__(self, model_name="pose_classifier", use_motion=True):
        # 加载模型
        bundle = load_bundle(model_name)
        self.model = bundle["model"]
        self.scaler = bundle.get("scaler")
        self.metrics = bundle.get("metrics", {})
        print(f" Model loaded: {model_name}")
        print(f"   Type: {self.metrics.get('model_type', 'unknown')}")
        if self.metrics.get('accuracy'):
            print(f"   Accuracy: {self.metrics.get('accuracy', 0):.1%}")

        # 特征提取器
        self.extractor = FeatureExtractor(use_motion=use_motion, smooth_window=0)

        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=MODEL_COMPLEXITY,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

        # 状态历史（用于平滑 + 时间线）
        self.state_history = collections.deque(maxlen=100)
        self.prob_history = collections.deque(maxlen=20)  # 概率平滑
        self._fall_alert_active = False
        self._fall_alert_start = 0.0

        # 跌倒运动学规则校验（防止坐姿被误判为 fall）
        # 记录最近帧的躯干角度和质心位置，用于检测"突降"特征
        self._torso_angle_history = collections.deque(maxlen=15)
        self._centroid_y_history = collections.deque(maxlen=15)
        self._fall_confirmed_frames = 0          # 连续确认跌倒的帧数
        self._FALL_CONFIRM_THRESHOLD = 5         # 需要连续 N 帧确认才触发告警
        self._FALL_TORSO_ANGLE_MIN = 45.0        # 躯干角 > 45° 才是躺姿/跌倒
        self._FALL_VELOCITY_MIN = 0.03           # 质心下坠速度阈值（帧间位移）

    def process_frame(self, frame_bgr):
        """处理一帧，返回 (annotated_frame, result_dict)"""
        h, w = frame_bgr.shape[:2]
        result = {
            "state": "unknown",
            "confidence": 0.0,
            "probabilities": {},
            "has_person": False,
            "alert_level": 0,
            "alert_message": "",
        }

        # 1. MediaPipe 推理
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        mp_results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        has_person = mp_results.pose_landmarks is not None
        result["has_person"] = has_person

        if not has_person:
            self.extractor.reset()
            self.state_history.append("unknown")
        else:
            # 2. 特征提取
            landmarks = landmarks_from_mediapipe(mp_results.pose_landmarks)
            fv = self.extractor.extract_with_motion(landmarks)
            X = fv.values.reshape(1, -1)

            # 3. 标准化 + 预测
            if self.scaler:
                X = self.scaler.transform(X)

            # 获取概率
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X)[0]
                # model.classes_ maps internal index -> actual class ID
                known_classes = getattr(self.model, 'classes_', list(range(len(proba))))
                cls_to_prob = {int(cls): float(p) for cls, p in zip(known_classes, proba)}
                best_idx = np.argmax(proba)
                pred_state = STATE_NAMES[int(known_classes[best_idx])]
                confidence = float(proba[best_idx])

                # Map to all 6 state names
                for i, name in enumerate(STATE_NAMES):
                    result["probabilities"][name] = cls_to_prob.get(i, 0.0)
            else:
                pred_idx = self.model.predict(X)[0]
                pred_state = STATE_NAMES[int(pred_idx)] if int(pred_idx) < len(STATE_NAMES) else "unknown"
                confidence = 1.0

            # 4. 概率平滑（减少闪烁）
            self.prob_history.append(pred_state)
            smoothed_state = self._smooth_prediction()

            # ---- 运动学规则校验：防止坐姿/慢动作被误判为 fall ----
            # 记录当前帧的躯干角度 & 质心高度
            torso_angle = fv.torso_features[3]   # 躯干角
            centroid_y = fv.torso_features[1]     # 质心高度
            self._torso_angle_history.append(torso_angle)
            self._centroid_y_history.append(centroid_y)

            # 计算运动学特征
            angular_velocity = 0.0
            centroid_velocity = 0.0
            if len(self._torso_angle_history) >= 3:
                # 帧间角度变化率（最近3帧平均）
                recent_angles = list(self._torso_angle_history)[-3:]
                angular_velocity = abs(recent_angles[-1] - recent_angles[-3]) / 3.0
            if len(self._centroid_y_history) >= 3:
                recent_centroids = list(self._centroid_y_history)[-3:]
                centroid_velocity = abs(recent_centroids[-1] - recent_centroids[-3]) / 3.0

            # 跌倒运动学验证:
            #   - 躯干角 > 阈值（人已经倾倒）
            #   - 角度变化快（快速倒下 vs. 慢速躺下/坐下）
            #   - 质心下坠速度快（跌倒的典型特征）
            is_fall_physics = (
                torso_angle > self._FALL_TORSO_ANGLE_MIN
                and angular_velocity > 1.5       # 角度快速变化
                and centroid_velocity > self._FALL_VELOCITY_MIN
            )

            # 如果模型预测 fall 但运动学不匹配 → 降级为 sitting/lying
            if smoothed_state == "fall" and not is_fall_physics:
                # 使用躯干角判断是 sitting 还是 lying
                if torso_angle > 40:
                    smoothed_state = "lying"
                else:
                    smoothed_state = "sitting"
                result["confidence"] *= 0.5  # 降低置信度

            # 如果模型预测 sitting/lying 但运动学符合 fall → 可能是漏检（保守处理）
            if smoothed_state in ("sitting", "lying") and is_fall_physics:
                if centroid_velocity > 0.05:  # 非常快速的运动
                    smoothed_state = "fall"

            result["state"] = smoothed_state
            result["confidence"] = confidence

            self.state_history.append(smoothed_state)

            # 5. 告警判断（需要连续确认帧，避免闪烁误报）
            if smoothed_state == "fall":
                self._fall_confirmed_frames += 1
                if self._fall_confirmed_frames >= self._FALL_CONFIRM_THRESHOLD:
                    self._fall_alert_active = True
                    self._fall_alert_start = time.time()
                    result["alert_message"] = " 跌倒检测！"
            else:
                self._fall_confirmed_frames = max(0, self._fall_confirmed_frames - 1)

            # 跌倒告警持续3秒后消除
            if self._fall_alert_active and time.time() - self._fall_alert_start > 3.0:
                self._fall_alert_active = False
                self._fall_confirmed_frames = 0

            alert_level = STATE_ALERT_LEVEL.get(smoothed_state, 0)
            result["alert_level"] = alert_level

            if smoothed_state == "long_sit":
                # 检查久坐持续时间
                sit_count = sum(1 for s in list(self.state_history)[-30:] if s in ("sitting", "long_sit"))
                if sit_count >= 25:  # 约 25/30 帧 = 久坐
                    result["alert_message"] = " 久坐提醒"
                    result["state"] = "long_sit"

        # 6. 绘制 UI
        annotated = self._draw_overlay(frame_bgr, mp_results, result)

        return annotated, result

    def _smooth_prediction(self):
        """概率平滑：取最近 N 帧的众数"""
        if len(self.prob_history) < 3:
            return self.prob_history[-1]
        recent = list(self.prob_history)[-7:]
        from collections import Counter
        return Counter(recent).most_common(1)[0][0]

    def reset(self):
        """重置推理状态"""
        self.extractor.reset()
        self.state_history.clear()
        self.prob_history.clear()
        self._torso_angle_history.clear()
        self._centroid_y_history.clear()
        self._fall_alert_active = False
        self._fall_confirmed_frames = 0

    def close(self):
        self.pose.close()

    # ------------------------------------------------------------------
    # UI 绘制
    # ------------------------------------------------------------------
    def _draw_overlay(self, frame, mp_results, result):
        h, w = frame.shape[:2]
        state = result["state"]
        conf = result["confidence"]
        alert_level = result["alert_level"]
        probs = result.get("probabilities", {})

        # --- 骨架（仅绘制身体关键点 11-32，排除面部 0-10） ---
        if mp_results.pose_landmarks:
            landmarks = mp_results.pose_landmarks.landmark
            h, w = frame.shape[:2]
            # 绘制身体连线
            for start_idx, end_idx in self.mp_pose.POSE_CONNECTIONS:
                if start_idx >= 11 and end_idx >= 11:  # 只画身体连线
                    x1, y1 = int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h)
                    x2, y2 = int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 245, 255), 2)
            # 绘制身体关键点
            for i in range(11, 33):
                lm = landmarks[i]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        # --- 跌倒告警闪烁 ---
        if self._fall_alert_active:
            elapsed = time.time() - self._fall_alert_start
            if int(elapsed * 4) % 2 == 0:  # 闪烁
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                # 大字告警
                cv2.putText(frame, " FALL DETECTED! ",
                            (w // 2 - 250, h // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)

        # --- 顶部状态栏 ---
        bar_h = 85
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
        frame = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)

        # 状态 + 置信度
        color = STATE_COLORS.get(state, (255, 255, 255))
        cv2.putText(frame, f"{state.upper()}", (15, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
        cv2.putText(frame, f"{conf:.1%}", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 告警级别指示器
        alert_color = ALERT_LEVEL_COLORS.get(alert_level, (128, 128, 128))
        alert_labels = ["NORMAL", "LOW", "MEDIUM", "HIGH"]
        alert_text = alert_labels[min(alert_level, 3)]
        cv2.putText(frame, f"ALERT: {alert_text}", (w - 250, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2)

        # 告警消息
        if result.get("alert_message"):
            cv2.putText(frame, result["alert_message"], (w - 250, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- 底部状态时间线 ---
        timeline_y = h - 40
        timeline_h = 30
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (0, timeline_y), (w, timeline_y + timeline_h),
                      (20, 20, 20), -1)
        frame = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)

        if self.state_history:
            history = list(self.state_history)
            segment_w = w / len(history) if history else 0
            for i, s in enumerate(history):
                x_start = int(i * segment_w)
                x_end = int((i + 1) * segment_w)
                seg_color = STATE_COLORS.get(s, (128, 128, 128))
                # 调暗非当前帧
                if i < len(history) - 5:
                    seg_color = tuple(int(c * 0.4) for c in seg_color)
                cv2.rectangle(frame,
                              (x_start, timeline_y + 2),
                              (x_end, timeline_y + timeline_h - 2),
                              seg_color, -1)

        # --- 无人检测提示 ---
        if not result["has_person"]:
            cv2.putText(frame, "No person detected",
                        (w // 2 - 120, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)

        return frame


# ============================================================
# 主循环
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="HuLing - Real-time Inference")
    parser.add_argument("--model", type=str, default="pose_classifier",
                        help="Model name (corresponds to .joblib file in models/)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default 0)")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode (print only)")
    parser.add_argument("--save-video", type=str, default=None,
                        help="Save inference video path")
    parser.add_argument("--no-motion", action="store_true",
                        help="Disable motion features (single-frame inference)")
    args = parser.parse_args()

    # 初始化引擎
    engine = InferenceEngine(
        model_name=args.model,
        use_motion=not args.no_motion,
    )

    # 打开摄像头（Windows 上使用 DSHOW 后端更稳定）
    if os.name == 'nt':
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f" Cannot open camera (index {args.camera})")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0:
        actual_fps = 30

    # 视频保存
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            args.save_video, fourcc, actual_fps, (actual_w, actual_h)
        )
        print(f" Saving video: {args.save_video}")

    # 性能统计
    fps_counter = 0
    fps_timer = time.time()
    fps = 0.0
    inference_times = collections.deque(maxlen=30)

    # 中文终端输出（不受编码限制）
    print(f"\n{'=' * 60}")
    print(f"  HuLing - Real-time Inference")
    print(f"  Press Q to quit | Press R to reset")
    print(f"{'=' * 60}\n")

    # 预先创建窗口（使用英文标题避免 Windows OpenCV 中文乱码）
    WINDOW_NAME = "HuLing - Real-time Inference"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("[DEBUG] Window created, starting capture...")

    # 摄像头热身（丢弃前几帧不稳定数据）
    print("[DEBUG] Camera warming up...")
    for i in range(10):
        s, _ = cap.read()
        if not s:
            print(f"[DEBUG] Warmup frame {i+1} read failed!")
            break
        time.sleep(0.01)
    print("[DEBUG] Warmup done, entering main loop")


    def _check_terminal_key():
        """检查终端按键（Windows msvcrt / Unix select）"""
        if os.name == 'nt' and HAS_MSVCRT:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                return ch
        else:
            # Unix: 尝试用 select 非阻塞读取 stdin
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                return ch.encode() if isinstance(ch, str) else ch
        return None

    try:
        frame_count = 0
        while True:
            success, frame = cap.read()
            if frame_count <= 2:
                print(f"[DEBUG] Frame {frame_count+1}: success={success}, "
                      f"frame={'OK' if frame is not None else 'None'}")
            if not success:
                print(f" Frame read failed (frame_count={frame_count})")
                time.sleep(2)  # 停2秒让人看到错误信息
                break

            frame_count += 1

            # 镜像翻转
            frame = cv2.flip(frame, 1)

            # 推理计时
            try:
                t0 = time.time()
                annotated, result = engine.process_frame(frame)
                inference_time = (time.time() - t0) * 1000  # ms
                inference_times.append(inference_time)
            except Exception as e:
                print(f"\n[ERROR] process_frame 异常: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
                break

            # FPS
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            # 打印结果（无头模式）
            if args.no_display:
                if result["has_person"]:
                    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
                          f"{result['state']:10s} | "
                          f"conf={result['confidence']:.2f} | "
                          f"alert={result['alert_level']} | "
                          f"infer={inference_time:.0f}ms",
                          end="", flush=True)

            # 叠加性能信息
            avg_infer = np.mean(inference_times) if inference_times else 0
            cv2.putText(annotated,
                        f"{fps:.0f} fps | infer {avg_infer:.0f}ms",
                        (15, annotated.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            # 显示
            if not args.no_display:
                if frame_count <= 2:
                    print(f"[DEBUG] Frame {frame_count} imshow...")
                cv2.imshow(WINDOW_NAME, annotated)

            # 保存视频
            if video_writer:
                video_writer.write(annotated)

            # ===== 键盘处理（双通道：OpenCV 窗口 + 终端） =====
            # 通道1: OpenCV 窗口按键（窗口有焦点时）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                engine.reset()
                print("\n[OK] Inference state reset (via OpenCV window)")

            # 通道2: 终端按键（窗口失焦时也能响应）
            term_key = _check_terminal_key()
            if term_key is not None:
                ch = term_key.lower() if isinstance(term_key, bytes) else term_key
                if isinstance(ch, bytes):
                    ch = ch.decode('ascii', errors='ignore').lower()
                else:
                    ch = str(ch).lower()
                if ch in ('q', '\x1b'):  # Q 或 ESC
                    break
                elif ch == 'r':
                    engine.reset()
                    print("\n[OK] Inference state reset (via terminal)")

    except KeyboardInterrupt:
        print("\n\n[STOP] User interrupted")
    except Exception as e:
        print(f"\n[FATAL] Main loop exception: {e}")
        import traceback
        traceback.print_exc()
        print("\nWindow will close in 10 seconds...")
        time.sleep(10)

    finally:
        engine.close()
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

    if args.no_display:
        print()  # newline

    print(f"\n Inference finished")
    print(f"   Avg inference time: {np.mean(inference_times):.1f}ms/frame"
          if inference_times else "")


if __name__ == "__main__":
    main()
