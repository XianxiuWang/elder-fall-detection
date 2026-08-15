#!/usr/bin/env python3
"""
cv_client.py — 步态数据客户端
=============================
从摄像头实时提取步态参数，周期性 POST 到 FastAPI 后端接口。
如需独立测试，可附带启动本地 Mock 服务器。

用法:
  python cv_client.py                           # USB 摄像头 (ID 0)，推送到默认后端
  python cv_client.py --camera 1                # USB 摄像头 ID 1
  python cv_client.py --rtsp rtsp://...         # RTSP 网络摄像头 (萤石等)
  python cv_client.py --ezviz-ip 192.168.1.100 --ezviz-code ABCDEF  # 萤石 C6c 一键连接
  python cv_client.py --mock-server             # 启动本地 Mock 服务器 + 客户端
  python cv_client.py --api-url http://x.x.x.x:8000  # 自定义后端地址
  python cv_client.py --save-video demo.mp4     # 保存标注视频

接口定义（对应《AI智能居家监护系统 API规范.md V1.0》4.3.2）:
  POST /api/v1/gait-data/
  {
    "elderly_id": 1,
    "walking_speed": 1.2,     # 行走速度
    "step_length": 0.65,      # 步长
    "body_sway": 5,           # 身体晃动幅度
    "balance_score": 95       # 平衡评分 (0-100)
  }

依赖: numpy, opencv-python, mediapipe, xgboost, requests
"""
import os, sys, time, json, pickle, argparse
import threading, warnings
from collections import deque
from datetime import datetime, timezone, timedelta

# 静音 mediapipe/protobuf 版本警告
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
os.environ["GLOG_minloglevel"] = "2"

import numpy as np
import cv2
import requests

# ─── 复用 fall_inference.py 的核心模块 ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fall_inference import (EnhancedFeatureExtractor, TemporalFilter,
                                 load_model, WINDOW_SIZE, WINDOW_STRIDE,
                                 SKIP_FRAMES, MODEL_COMPLEXITY, CLASS_NAMES,
                                 FALL_ID, DEFAULT_MODEL)
except ImportError:
    print("[ERROR] 无法导入 fall_inference.py，请确认在同一目录下")
    sys.exit(1)

# ─── 配置 ───
CST = timezone(timedelta(hours=8))
DEFAULT_API_URL = "http://127.0.0.1:8000"
POST_INTERVAL_SEC = 5.0          # 每 5 秒推送一次步态数据
ELDERLY_ID = 1                    # 老人编号（暂固定）

# 人体参考尺寸（MediaPipe 归一化坐标 → 近似物理值）
# 假设躯干长度 0.12 归一化 → 约 45cm 实际
NORMALIZED_TORSO_TO_CM = 375.0   # 1.0 归一化 Y → ~375cm (画面高度=2m / 0.53≈375)
# 行走速度 = 重心归一化位移/秒 × 归一化→cm 换算
PIXEL_TO_CM = 180.0               # 1.0 归一化 Y → 约 180cm (站立人高度)

# ─── RTSP / 网络摄像头 ───
RTSP_RECONNECT_DELAY = 3.0        # RTSP 断流后重连间隔 (秒)
RTSP_MAX_RECONNECT_ATTEMPTS = 10  # 连续重连失败多少次后放弃 (0 = 无限)
RTSP_BUFFER_SIZE = 1              # FFMPEG 缓冲帧数，越小延迟越低 (1 = 实时)


def build_ezviz_rtsp_url(ip, code, channel=1, stream="main"):
    """
    根据萤石 (EZVIZ) 摄像头 IP + 设备验证码 生成 RTSP 地址。
    
    适用: 萤石 C6c / C6CN 等系列网络摄像头。
    
    参数:
      ip      : 摄像头局域网 IP (如 192.168.1.100)
      code    : 设备底部标签上的 6 位大写字母验证码
      channel : 通道号 (默认 1)
      stream  : "main" 主码流 (高清, 占带宽) / "sub" 子码流 (流畅, 推荐用于推理)
    
    返回:
      "rtsp://admin:CODE@IP:554/h264/ch1/main/av_stream"
    
    注意:
      - 需在「萤石云视频」App 中开启: 设备设置 → 本地服务设置 → 开启 RTSP。
      - 默认端口 554，如路由器端口映射过需自行替换。
    """
    return f"rtsp://admin:{code}@{ip}:554/h264/ch{channel}/{stream}/av_stream"


def open_video_source(source):
    """
    打开视频源，返回 (VideoCapture, 描述字符串)。
    
    source:
      int  → 本地摄像头 ID
      str  → RTSP / 网络流 URL (如 rtsp://..., http://...)
    """
    if isinstance(source, str):
        desc = source
        # 用 FFMPEG 后端打开网络流，减小缓冲以降低延迟
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        # 缓冲设为 1 帧，追求最低延迟 (OpenCV 4.x)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, RTSP_BUFFER_SIZE)
        except Exception:
            pass
    else:
        desc = f"本地摄像头 {source}"
        cap = cv2.VideoCapture(source)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap, desc


def compute_gait_params(window, effective_fps, window_size=30):
    """
    从关键点窗口计算步态参数。
    
    window: (30, 33, 3) ndarray [frame, landmark, xyz]
    effective_fps: 实际采样帧率
    
    返回:
      {
        "walking_speed": float,  # 行走速度
        "step_length": float,    # 步长
        "body_sway": float,      # 身体晃动幅度
        "balance_score": float,  # 平衡评分 (0-100)
      }
    """
    T = len(window)
    if T < 3:
        return {"walking_speed": 0.0, "step_length": 0.0, "body_sway": 0.0, "balance_score": 50.0}
    
    x = window[:, :, 0]
    y = window[:, :, 1]
    
    # ── 1. 行走速度 ──
    # 重心: 双肩+双髋+双膝 六点 Y 均值
    centroid_y = np.mean(y[:, [11, 12, 23, 24, 25, 26]], axis=1)
    # 帧间位移 → 累积总位移 → 平均速度 (归一化/秒)
    frame_disp = np.abs(np.diff(centroid_y))
    total_disp = float(np.sum(frame_disp))
    norm_speed = total_disp / (T / effective_fps)  # 归一化坐标/秒
    speed_cms = norm_speed * PIXEL_TO_CM            # → cm/s
    
    # ── 2. 步长 ──
    # 左右脚踝 Y 坐标差 → 跨步距离
    left_ankle_y = y[:, 27]
    right_ankle_y = y[:, 28]
    ankle_diff = np.abs(left_ankle_y - right_ankle_y)
    # 步长: 脚踝差值的 95 分位数 (过滤站立时小值)
    step_length = float(np.percentile(ankle_diff, 95)) * 100
    
    # ── 3. 身体晃动 ──
    # 双肩中点 X 坐标的 std → 侧向晃动量
    shoulder_mid_x = (x[:, 11] + x[:, 12]) / 2
    sway = float(np.std(shoulder_mid_x)) * 100
    
    # ── 4. 步态综合评分 (0-100) ──
    # 基于: 速度稳定性 + 晃动程度 + 步长规律性
    speed_consistency = 1.0 - min(1.0, float(np.std(frame_disp) / (np.mean(frame_disp) + 1e-6)))
    sway_penalty = min(1.0, sway / 15.0)           # 晃动 > 15 → 满分惩罚
    step_regularity = 1.0 - min(1.0, float(np.std(ankle_diff)) / (np.mean(ankle_diff) + 1e-6))
    
    gait_score = float(np.clip(
        (speed_consistency * 40 + (1.0 - sway_penalty) * 30 + step_regularity * 30),
        0, 100
    ))
    
    return {
        "walking_speed": round(speed_cms, 2),   # 注意: 当前为 cm/s，需与后端对齐单位(m/s?)
        "step_length": round(step_length, 2),
        "body_sway": round(sway, 1),
        "balance_score": round(gait_score, 1),
    }


class GaitDataClient:
    """步态数据推送客户端 — 线程安全"""
    
    def __init__(self, api_url=DEFAULT_API_URL, elderly_id=ELDERLY_ID,
                 interval=POST_INTERVAL_SEC):
        self.api_url = api_url.rstrip("/")
        self.elderly_id = elderly_id
        self.interval = interval
        self.last_post = 0.0
        self.lock = threading.Lock()
        self.stats = {"sent": 0, "errors": 0}
    
    def send_if_ready(self, gait_data):
        """如果超过间隔时间，发送步态数据到后端。线程安全。"""
        now = time.time()
        with self.lock:
            if now - self.last_post < self.interval:
                return False
            self.last_post = now
        
        # API 规范 V1.0 扁平结构：elderly_id + walking_speed/step_length/body_sway/balance_score
        payload = {"elderly_id": self.elderly_id}
        payload.update(gait_data)
        
        try:
            resp = requests.post(
                f"{self.api_url}/api/v1/gait-data/",
                json=payload,
                timeout=3.0,
            )
            if resp.status_code == 200:
                try:
                    ok = resp.json().get("code", 200) == 200
                except Exception:
                    ok = True
                if ok:
                    self.stats["sent"] += 1
                    return True
            self.stats["errors"] += 1
            return False
        except requests.exceptions.ConnectionError:
            # 后端未启动，静默跳过
            self.stats["errors"] += 1
            return False
        except Exception as e:
            self.stats["errors"] += 1
            return False


def run_realtime_with_client(model_path=DEFAULT_MODEL, source=0,
                              camera_fps=30.0, api_url=DEFAULT_API_URL,
                              save_video=None):
    """实时摄像头推理 + 步态数据推送。source: int=本地摄像头ID, str=RTSP/网络流URL"""
    import mediapipe as mp
    
    is_rtsp = isinstance(source, str)
    
    print("=" * 60)
    print("  AI 智能居家监护系统 — 实时演示")
    print("=" * 60)
    print(f"  后端 API: {api_url}/api/v1/gait-data/")
    print(f"  推送间隔: {POST_INTERVAL_SEC}s")
    print(f"  视频源: {'RTSP 网络流' if is_rtsp else '本地摄像头'}")
    print(f"  按 'q' 退出 | 按 's' 截图")
    print("=" * 60)
    
    # ── 加载模型 ──
    print("\n[1/3] 加载模型...", flush=True)
    model, scaler, bundle = load_model(model_path)
    extractor = EnhancedFeatureExtractor(window_size=WINDOW_SIZE)
    print(f"  ✅ {bundle.get('config', {}).get('version', 'V6')}, "
          f"{bundle['feature_dim']}维, "
          f"准确率 {bundle.get('metrics', {}).get('accuracy', '?')}",
          flush=True)
    
    # ── 打开摄像头 ──
    print("\n[2/3] 打开摄像头...", flush=True)
    cap, source_desc = open_video_source(source)
    if not cap.isOpened():
        print(f"  ❌ 无法打开视频源: {source_desc}")
        if is_rtsp:
            print("  提示: 请检查 RTSP 地址、摄像头是否开启 RTSP 服务、")
            print("        以及板子与摄像头是否在同一局域网内。")
        sys.exit(1)
    print(f"  ✅ {source_desc} 已就绪", flush=True)
    
    effective_fps = camera_fps / SKIP_FRAMES
    
    # ── MediaPipe Pose ──
    print("\n[3/3] 启动 MediaPipe Pose + 推理...", flush=True)
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4)
    
    keypoint_buffer = deque(maxlen=WINDOW_SIZE)
    filt = TemporalFilter(effective_fps=effective_fps)
    client = GaitDataClient(api_url=api_url)
    
    # 视频保存
    out_writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(save_video, fourcc, effective_fps, (640, 480))
        print(f"  📹 录制: {save_video}", flush=True)
    
    # 显示平滑: 标签必须连续 N 次相同才切换显示
    display_label = -1
    display_label_votes = 0
    DISPLAY_SMOOTH = 3  # 连续 3 次相同才切换
    frame_idx = 0
    processed = 0
    fall_alert = False
    screenshot_count = 0
    last_gait = {"walking_speed": 0.0, "step_length": 0.0, "body_sway": 0.0, "balance_score": 50.0}
    last_pose_result = None  # 缓存上一次的骨架，跳帧时复用
    # 显示缓存（所有帧统一绘制，消除闪烁）
    disp_label = "Initializing..."
    disp_color = (128, 128, 128)
    disp_gait = last_gait
    disp_sent = False
    disp_progress = 0.0
    disp_fall_border = False
    
    t0 = time.time()
    reconnect_fails = 0  # 连续读帧失败计数
    
    def try_reconnect():
        """释放并重开视频源 (RTSP 断流恢复)。返回 True 表示重连成功。"""
        nonlocal cap, reconnect_fails
        cap.release()
        time.sleep(RTSP_RECONNECT_DELAY)
        new_cap, _ = open_video_source(source)
        if new_cap.isOpened():
            cap = new_cap
            reconnect_fails = 0
            print(f"  ✅ 视频源已重连: {source_desc}", flush=True)
            return True
        return False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                reconnect_fails += 1
                print(f"  ⚠️ 视频流中断 (第 {reconnect_fails} 次)，尝试重连...",
                      flush=True)
                if is_rtsp:
                    if not try_reconnect():
                        if (RTSP_MAX_RECONNECT_ATTEMPTS > 0 and
                                reconnect_fails >= RTSP_MAX_RECONNECT_ATTEMPTS):
                            print("  ❌ 重连失败次数过多，退出。", flush=True)
                            break
                        print(f"  ⏳ 重连失败，{RTSP_RECONNECT_DELAY}s 后重试...",
                              flush=True)
                        continue
                else:
                    time.sleep(0.5)
                # 重置显示状态，避免残留旧骨架
                last_pose_result = None
                disp_label = "Reconnecting..."
                disp_color = (128, 128, 128)
                disp_fall_border = False
                continue
            
            if frame_idx % SKIP_FRAMES != 0:
                pass   # skip inference, fall through to unified draw
            else:
                # ── Processing (only on non-skip frames) ──
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_mp = pose.process(rgb)

                if results_mp.pose_landmarks:
                    last_pose_result = results_mp
                    kpts = np.array([[lm.x, lm.y, lm.visibility]
                                     for lm in results_mp.pose_landmarks.landmark],
                                    dtype=np.float32)
                    keypoint_buffer.append(kpts)

                    if len(keypoint_buffer) == WINDOW_SIZE:
                        processed += 1
                        window = np.array(keypoint_buffer, dtype=np.float32)
                        vec = extractor.extract_window(window)
                        vec_s = scaler.transform(vec.reshape(1, -1))
                        probs = model.predict_proba(vec_s)[0]
                        raw_label = int(np.argmax(probs))
                        filt_label, confidence = filt.update(probs, window)

                        last_gait = compute_gait_params(window, effective_fps)
                        disp_sent = client.send_if_ready(last_gait)

                        if filt_label == display_label:
                            display_label_votes += 1
                        else:
                            display_label_votes = 1
                        if display_label_votes >= DISPLAY_SMOOTH:
                            display_label = filt_label
                        shown_label = display_label if display_label >= 0 else filt_label

                        disp_label = f"{CLASS_NAMES[shown_label]} ({confidence:.2f})"
                        disp_gait = last_gait
                        disp_progress = 1.0

                        if filt_label == FALL_ID:
                            disp_color = (0, 0, 255)
                            disp_fall_border = True
                            if not fall_alert:
                                print(f"\n  🚨 检测到跌倒! frame={frame_idx}", flush=True)
                                fall_alert = True
                        elif filt_label == 3:
                            disp_color = (0, 255, 255)
                            disp_fall_border = False
                            fall_alert = False
                        elif filt_label == 5:
                            disp_color = (128, 128, 128)
                            disp_fall_border = False
                            fall_alert = False
                        else:
                            disp_color = (0, 255, 128)
                            disp_fall_border = False
                            fall_alert = False

                        for _ in range(WINDOW_STRIDE):
                            if keypoint_buffer:
                                keypoint_buffer.popleft()
                    else:
                        disp_progress = len(keypoint_buffer) / WINDOW_SIZE
                else:
                    last_pose_result = None
                    disp_label = "No person detected"
                    disp_color = (128, 128, 128)
                    disp_progress = 0.0
                    fall_alert = False

            # ═══════════════════════════════════════
            # Unified Draw — every frame, zero flicker
            # ═══════════════════════════════════════

            if last_pose_result and last_pose_result.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    last_pose_result.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 255, 128), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(255, 255, 255), thickness=1, circle_radius=1),
                )

            cv2.putText(frame, disp_label, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, disp_color, 3)

            if disp_fall_border:
                cv2.rectangle(frame, (0, 0), (639, 479), (0, 0, 255), 5)
                cv2.putText(frame, "!! FALL DETECTED !!", (120, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            y0 = 70
            gait_lines = [
                f"Speed: {disp_gait['walking_speed']:.1f} cm/s",
                f"Step:  {disp_gait['step_length']:.1f}",
                f"Sway:  {disp_gait['body_sway']:.1f}",
                f"Balance: {disp_gait['balance_score']:.0f}/100",
            ]
            for i, line in enumerate(gait_lines):
                cv2.putText(frame, line, (10, y0 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 200, 200), 1)

            if disp_sent:
                cv2.putText(frame, "📤 sent", (550, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)

            if disp_progress > 0 and disp_progress < 1.0:
                bar_x = int(540 * disp_progress)
                cv2.rectangle(frame, (50, 460), (590, 472), (80, 80, 80), -1)
                cv2.rectangle(frame, (50, 460), (50 + bar_x, 472),
                              (0, 255, 128), -1)
                cv2.putText(frame, f"Buffering... {len(keypoint_buffer)}/{WINDOW_SIZE}",
                            (55, 456), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (200, 200, 200), 1)

            elapsed = time.time() - t0
            fps_display = processed / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps_display:.1f} | Frame: {frame_idx}",
                        (10, 475), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (150, 150, 150), 1)

            if out_writer:
                out_writer.write(frame)
            
            # Display the frame
            cv2.imshow("AI Fall Detection — Real-time Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                screenshot_count += 1
                ss_path = f"screenshot_{screenshot_count:03d}.png"
                cv2.imwrite(ss_path, frame)
                print(f"  📸 截图已保存: {ss_path}", flush=True)
            
            frame_idx += 1
            
    except KeyboardInterrupt:
        print("\n  ⏹️  用户中断", flush=True)
    finally:
        pose.close()
        cap.release()
        if out_writer:
            out_writer.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  演示结束 — {elapsed:.0f}s, {processed} 帧")
        print(f"  步态推送: {client.stats['sent']} 次成功, "
              f"{client.stats['errors']} 次失败")
        print(f"{'='*60}")


def run_mock_server(port=8000):
    """启动本地 Mock 服务器，用于独立测试（无真实后端时）"""
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
    except ImportError:
        print("[ERROR] 无法导入 http.server")
        return
    
    class MockHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            data = json.loads(body)
            
            print(f"\n  📥 [{datetime.now(CST).strftime('%H:%M:%S')}] "
                  f"POST {self.path}")
            print(f"     walking_speed={data.get('walking_speed', '?')} | "
                  f"step={data.get('step_length', '?')} | "
                  f"body_sway={data.get('body_sway', '?')} | "
                  f"balance={data.get('balance_score', '?')}/100")
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(
                {"code": 200, "message": "success", "data": {}}).encode())
        
        def log_message(self, format, *args):
            pass  # 抑制 HTTP 日志
    
    server = HTTPServer(("0.0.0.0", port), MockHandler)
    print(f"\n  🖥️  Mock API 服务器已启动: http://localhost:{port}")
    print(f"  POST http://localhost:{port}/api/v1/gait-data/")
    print(f"  按 Ctrl+C 停止\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  ⏹️  Mock 服务器已停止")
        server.shutdown()


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="步态数据客户端 — 实时摄像头 + API 推送")
    
    parser.add_argument("--api-url", default=DEFAULT_API_URL,
                        help=f"后端 API 地址 (默认: {DEFAULT_API_URL})")
    parser.add_argument("--camera", type=int, default=0,
                        help="本地 USB 摄像头 ID (默认 0)")
    parser.add_argument("--rtsp", dest="rtsp_url",
                        help="RTSP / 网络流地址 (如 rtsp://admin:xxx@ip:554/...)")
    parser.add_argument("--ezviz-ip",
                        help="萤石摄像头局域网 IP，配合 --ezviz-code 自动拼接 RTSP 地址")
    parser.add_argument("--ezviz-code",
                        help="萤石设备底部 6 位大写字母验证码")
    parser.add_argument("--ezviz-stream", choices=["main", "sub"], default="sub",
                        help="萤石码流: main 高清 / sub 流畅 (默认 sub，推荐用于推理)")
    parser.add_argument("--camera-fps", type=float, default=30.0,
                        help="摄像头帧率 (默认 30)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"模型路径")
    parser.add_argument("--save-video",
                        help="保存标注视频路径")
    parser.add_argument("--mock-server", action="store_true",
                        help="同时启动本地 Mock API 服务器")
    parser.add_argument("--mock-port", type=int, default=8000,
                        help="Mock 服务器端口 (默认 8000)")
    parser.add_argument("--no-push", action="store_true",
                        help="不推送数据到后端（仅本地演示）")
    
    args = parser.parse_args()
    
    # ── 解析视频源 ──
    # 优先级: --rtsp > --ezviz-ip+code > --camera
    if args.rtsp_url:
        source = args.rtsp_url
        print(f"  🎥 使用 RTSP 流: {args.rtsp_url}")
    elif args.ezviz_ip:
        if not args.ezviz_code:
            print("  ❌ 使用 --ezviz-ip 时必须同时提供 --ezviz-code (设备验证码)")
            sys.exit(1)
        source = build_ezviz_rtsp_url(args.ezviz_ip, args.ezviz_code,
                                      stream=args.ezviz_stream)
        # 隐藏验证码，仅打印不含密码的地址
        print(f"  🎥 使用萤石 RTSP 流: {args.ezviz_ip}:554 "
              f"(stream={args.ezviz_stream})")
    else:
        source = args.camera
    
    if args.mock_server:
        # 在后台线程启动 Mock 服务器
        server_thread = threading.Thread(
            target=run_mock_server,
            args=(args.mock_port,),
            daemon=True,
        )
        server_thread.start()
        time.sleep(1)  # 等待服务器就绪
        api_url = f"http://127.0.0.1:{args.mock_port}"
    else:
        api_url = args.api_url
    
    if args.no_push:
        api_url = "http://127.0.0.1:1"  # 不存在的地址，静默失败
    
    run_realtime_with_client(
        model_path=args.model,
        source=source,
        camera_fps=args.camera_fps,
        api_url=api_url,
        save_video=args.save_video,
    )
