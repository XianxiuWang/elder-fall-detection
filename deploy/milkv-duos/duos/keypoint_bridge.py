"""
keypoint_bridge.py — 从 MediaPipe/其他姿态估计 提取关键点，发送给 Duo S

支持两种模式:
  1. 文件模式: 将关键点保存为文本文件，复制到 SD 卡供 Duo S 离线推理
  2. 网络模式: 通过 UDP/TCP 实时发送关键点到 Duo S

用法:
    python keypoint_bridge.py --input camera --output file --path keypoints.txt
    python keypoint_bridge.py --input camera --output udp --host 192.168.1.100 --port 8888
    python keypoint_bridge.py --input video.mp4 --output file
"""
import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("[WARN] MediaPipe not installed. Install with: pip install mediapipe")
    print("[WARN] Using OpenCV DNN MoveNet fallback...")


# ============================================================
# MediaPipe Pose → 33 keypoints (matching HuLing format)
# ============================================================
KEYPOINT_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# MoveNet (17 keypoints) → MediaPipe (33 keypoints) mapping
# We'll interpolate missing facial landmarks
MOVENET_TO_MEDIAPIPE_MAP = {
    # MoveNet index → MediaPipe index
    0:  0,   # nose → nose
    1:  2,   # left_eye → left_eye (approximate)
    2:  5,   # right_eye → right_eye (approximate)
    3:  7,   # left_ear → left_ear
    4:  8,   # right_ear → right_ear
    5:  11,  # left_shoulder → left_shoulder
    6:  12,  # right_shoulder → right_shoulder
    7:  13,  # left_elbow → left_elbow
    8:  14,  # right_elbow → right_elbow
    9:  15,  # left_wrist → left_wrist
    10: 16,  # right_wrist → right_wrist
    11: 23,  # left_hip → left_hip
    12: 24,  # right_hip → right_hip
    13: 25,  # left_knee → left_knee
    14: 26,  # right_knee → right_knee
    15: 27,  # left_ankle → left_ankle
    16: 28,  # right_ankle → right_ankle
}


class MediaPipeExtractor:
    """Use MediaPipe Pose to get 33 keypoints"""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks is None:
            return None
        kps = []
        for lm in results.pose_landmarks.landmark:
            kps.append({
                "x": lm.x, "y": lm.y, "z": lm.z,
                "v": lm.visibility
            })
        return kps

    def close(self):
        self.pose.close()


class MoveNetExtractor:
    """Use OpenCV DNN MoveNet (lighter, runs on Duo S eventually)"""
    def __init__(self, model_path="movenet_lightning.tflite"):
        self.model_path = model_path
        self.input_size = (192, 192)  # MoveNet Lightning
        self.model = None
        self._load_model()

    def _load_model(self):
        if not Path(self.model_path).exists():
            print(f"[INFO] MoveNet model not found at {self.model_path}")
            print(f"[INFO] Download from: https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4")
            self.model = None
            return
        self.model = cv2.dnn.readNetFromTensorflow(self.model_path)

    def extract(self, frame_bgr):
        if self.model is None:
            return None

        h, w = frame_bgr.shape[:2]
        # Resize to 192x192
        input_img = cv2.resize(frame_bgr, self.input_size)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        self.model.setInput(input_img)
        output = self.model.forward()

        # MoveNet output: (1, 1, 1, 17, 3) -> [y, x, confidence]
        keypoints = output[0][0][0]

        # Convert 17 MoveNet keypoints → 33 MediaPipe-style keypoints
        kps_33 = []
        for i in range(33):
            kps_33.append({"x": 0.0, "y": 0.0, "z": 0.0, "v": 0.0})

        for mn_idx, (mp_idx) in MOVENET_TO_MEDIAPIPE_MAP.items():
            y, x, conf = keypoints[mn_idx]
            kps_33[mp_idx] = {
                "x": float(x / w),
                "y": float(y / h),
                "z": 0.0,
                "v": float(conf)
            }

        # Interpolate face landmarks (0-10) from nose + eyes + ears
        # For simplicity, copy nose to all face points
        nose = kps_33[0]
        face_indices = [1, 2, 3, 4, 5, 6, 9, 10]  # eyes + mouth
        for fi in face_indices:
            if kps_33[fi]["v"] < 0.1:
                kps_33[fi] = dict(nose)
                kps_33[fi]["v"] = nose["v"] * 0.5

        # Interpolate hands (17-22) from wrists + elbows
        for side, wrist_idx, elbow_idx, pinky_idx, index_idx, thumb_idx in [
            ("left", 15, 13, 17, 19, 21),
            ("right", 16, 14, 18, 20, 22),
        ]:
            wrist = kps_33[wrist_idx]
            elbow = kps_33[elbow_idx]
            for idx in [pinky_idx, index_idx, thumb_idx]:
                kps_33[idx] = {
                    "x": wrist["x"] + (wrist["x"] - elbow["x"]) * 0.2,
                    "y": wrist["y"] + (wrist["y"] - elbow["y"]) * 0.2,
                    "z": wrist["z"],
                    "v": wrist["v"] * 0.5
                }

        # Interpolate feet (29-32) from ankles
        for side, ankle_idx, heel_idx, foot_idx in [
            ("left", 27, 29, 31),
            ("right", 28, 30, 32),
        ]:
            ankle = kps_33[ankle_idx]
            for idx in [heel_idx, foot_idx]:
                kps_33[idx] = {
                    "x": ankle["x"],
                    "y": ankle["y"] + 0.02,
                    "z": ankle["z"],
                    "v": ankle["v"] * 0.5
                }

        return kps_33

    def close(self):
        pass


# ============================================================
# Output writers
# ============================================================
class FileWriter:
    """Save keypoints to text file (compatible with Duo S huling_demo)"""
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')

    def write(self, kps, frame_id=None):
        for kp in kps:
            # Format: {"x":0.5,"y":0.3,"z":0.0,"v":0.95}
            self.f.write(
                '{"x":%.6f,"y":%.6f,"z":%.6f,"v":%.6f}\n' %
                (kp["x"], kp["y"], kp["z"], kp["v"])
            )
        self.f.write("---\n")
        self.f.flush()

    def close(self):
        self.f.close()


class UDPWriter:
    """Send keypoints over UDP to Duo S"""
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (host, port)

    def write(self, kps, frame_id=None):
        # Pack: 4 bytes frame_id + 33 * 4 floats (x,y,z,v) = 4 + 528 = 532 bytes
        data = struct.pack('<I', frame_id or 0)
        for kp in kps:
            data += struct.pack('<ffff', kp["x"], kp["y"], kp["z"], kp["v"])
        self.sock.sendto(data, self.addr)

    def close(self):
        self.sock.close()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Keypoint bridge for Duo S")
    parser.add_argument("--input", type=str, default="camera",
                        help="camera / video.mp4 / image.jpg")
    parser.add_argument("--output", type=str, default="file",
                        choices=["file", "udp", "stdout"])
    parser.add_argument("--path", type=str, default="keypoints.txt")
    parser.add_argument("--host", type=str, default="192.168.1.100")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--backend", type=str, default="mediapipe",
                        choices=["mediapipe", "movenet"])
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    # Select extractor
    if args.backend == "mediapipe" and HAS_MEDIAPIPE:
        extractor = MediaPipeExtractor()
        print("[INFO] Using MediaPipe Pose (33 keypoints)")
    else:
        extractor = MoveNetExtractor()
        print("[INFO] Using MoveNet Lightning (17→33 keypoints)")

    # Select output
    if args.output == "file":
        writer = FileWriter(args.path)
        print(f"[INFO] Saving keypoints to: {args.path}")
    elif args.output == "udp":
        writer = UDPWriter(args.host, args.port)
        print(f"[INFO] Sending to Duo S at {args.host}:{args.port}")
    else:
        writer = None

    # Open input
    if args.input == "camera":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open input: {args.input}")
        return

    print("[INFO] Press 'q' to quit, 's' to skip frame")
    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            kps = extractor.extract(frame)

            if kps and writer:
                writer.write(kps, frame_id)

                # Draw skeleton for preview
                h, w = frame.shape[:2]
                for kp in kps:
                    if kp["v"] > 0.5:
                        cx, cy = int(kp["x"] * w), int(kp["y"] * h)
                        cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
            elif kps and writer is None:
                # stdout mode: print JSON
                print(json.dumps(kps))

            cv2.putText(frame, f"Frame {frame_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Keypoint Bridge", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    finally:
        extractor.close()
        if writer:
            writer.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"[INFO] Done. {frame_id} frames processed.")
    if args.output == "file":
        print(f"[INFO] Keypoints saved to: {args.path}")
        print(f"[INFO] Copy to Duo S SD card and run:")
        print(f"        ./huling_demo < {args.path}")


if __name__ == "__main__":
    main()
