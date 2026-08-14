"""
护龄 — Web 监控后端
====================
Flask + Socket.IO 实时数据推送服务器

功能：
  · REST API: 接收开发板 POST 的传感器数据 + 状态数据
  · WebSocket: 实时推送数据到所有连接的浏览器
  · MJPEG 视频流: 摄像头画面实时推流
  · 告警管理: 告警记录存储 + 历史查询

开发板对接方式（二选一）：
  方式A — REST API（简单，推荐 ESP32）:
    POST http://<服务器IP>:5000/api/data
    Content-Type: application/json
    {
      "device_id": "huling_001",
      "state_id": 1,
      "state_name": "坐着/休息",
      "heart_rate": 72,
      "spo2": 98,
      "temperature": 36.5,
      "battery": 85,
      "timestamp": 1714788000000
    }

  方式B — WebSocket（低延迟，推荐 Python端）:
    连接 ws://<服务器IP>:5000
    发送 JSON: {"type": "data", "payload": {...}}

用法：
    python server.py                  # 启动服务
    python server.py --host 0.0.0.0   # 允许局域网访问
    python server.py --port 8080      # 自定义端口
"""

import os
import sys
import json
import time
import queue
import threading
import argparse
from pathlib import Path
from datetime import datetime
from collections import deque

import cv2
import numpy as np
from flask import (
    Flask, render_template, request, jsonify,
    Response, send_from_directory
)
from flask_socketio import SocketIO, emit

# 添加父目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ============================================================
# 工具函数 — 白天时间计算
# ============================================================

def daytime_seconds_between(start: datetime, end: datetime) -> float:
    """
    计算两个时间点之间属于白天（config.INACTIVITY_ALERT 中定义的时间段）的秒数。

    支持跨天场景：例如从 20:00 到第二天 10:00，
    只计算其中属于白天的部分（当天 20:00-22:00 + 第二天 6:00-10:00）。

    Args:
        start: 起始时间
        end: 结束时间

    Returns:
        属于白天的秒数（浮点数）
    """
    if start >= end:
        return 0.0

    DAYTIME_START = config.INACTIVITY_ALERT["daytime_start"]
    DAYTIME_END = config.INACTIVITY_ALERT["daytime_end"]
    DAYTIME_DURATION = (DAYTIME_END - DAYTIME_START) * 3600  # 每天白天的秒数

    total = 0.0
    current = start

    # 处理第一天（可能是不完整的白天/黑夜段）
    hour = current.hour
    if DAYTIME_START <= hour < DAYTIME_END:
        # 当前在白天 → 计算到今天白天结束
        day_end = current.replace(hour=DAYTIME_END, minute=0, second=0, microsecond=0)
        segment = min(day_end, end)
        total += (segment - current).total_seconds()
        current = segment
    elif hour >= DAYTIME_END:
        # 当前在晚上 → 跳到第二天白天开始
        current = (current + timedelta(days=1)).replace(
            hour=DAYTIME_START, minute=0, second=0, microsecond=0
        )
    else:
        # 当前在凌晨 → 跳到今天白天开始
        current = current.replace(hour=DAYTIME_START, minute=0, second=0, microsecond=0)

    if current >= end:
        return total

    # 确保 current 在白天起点（刚处理完第一段后可能在黑夜边界如22:00）
    if current.hour >= DAYTIME_END or current.hour < DAYTIME_START:
        if current.hour >= DAYTIME_END:
            # 当前在晚上 → 跳到第二天白天开始
            current = (current + timedelta(days=1)).replace(
                hour=DAYTIME_START, minute=0, second=0, microsecond=0
            )
        else:
            # 当前在凌晨 → 跳到今天白天开始
            current = current.replace(hour=DAYTIME_START, minute=0, second=0, microsecond=0)

    if current >= end:
        return total

    # 剩余部分：按整天计算 + 最后不完整的一天
    full_days = int((end - current).total_seconds() / 86400)
    if full_days > 0:
        total += full_days * DAYTIME_DURATION
        current += timedelta(days=full_days)

    # 处理最后不完整的一天
    if current < end:
        remainder_day_end = current.replace(
            hour=DAYTIME_END, minute=0, second=0, microsecond=0
        )
        segment = min(remainder_day_end, end)
        if segment > current:
            total += (segment - current).total_seconds()

    return total


# ============================================================
# Flask 应用初始化
# ============================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'huling-2026'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ============================================================
# 全局状态存储
# ============================================================

class SystemState:
    """内存中的系统状态（开发板最新一帧的数据）"""

    def __init__(self):
        self.device_id = "huling_001"
        self.state_id = 6          # 初始默认无人
        self.state_name = "无人活动"
        self.confidence = 0.0
        self.heart_rate = None
        self.spo2 = None
        self.temperature = None
        self.battery = None
        self.last_update = 0
        self.is_online = False
        self.alert_level = "🟢 正常"
        self.alert_color = "green"

        # 告警历史（最近100条）
        self.alert_history = deque(maxlen=100)

        # 状态时间线（最近24h，每分钟一个点）
        self.state_timeline = deque(maxlen=1440)

        # ── 无人活动计时（白天累计，黑夜不计时） ──
        self.inactive_start = None           # 开始无人活动的时刻 (datetime)
        self.daytime_inactive_seconds = 0.0  # 白天累计无人活动秒数
        self.last_inactive_check = None      # 上次计时检查的时刻 (datetime)
        self._inactive_alert_sent = False    # 本次无人活动中是否已发送过告警

        # 心率/血氧/体温历史（用于画趋势图）
        self.vital_history = {
            'timestamps': deque(maxlen=1440),
            'heart_rate': deque(maxlen=1440),
            'spo2': deque(maxlen=1440),
            'temperature': deque(maxlen=1440),
        }

    def update_from_api(self, data: dict):
        """从 REST API 更新状态"""
        self.device_id = data.get('device_id', self.device_id)
        self.state_id = data.get('state_id', self.state_id)
        self.state_name = data.get('state_name', config.STATE_NAMES.get(self.state_id, '未知'))
        self.confidence = data.get('confidence', 0.0)
        self.heart_rate = data.get('heart_rate')
        self.spo2 = data.get('spo2')
        self.temperature = data.get('temperature')
        self.battery = data.get('battery')
        self.last_update = time.time()
        self.is_online = True

        # 告警级别
        alert_level, alert_color = config.ALERT_LEVEL.get(self.state_id, ('', ''))
        self.alert_level = alert_level
        self.alert_color = alert_color

        # 更新生命体征历史
        now = datetime.now().strftime("%H:%M")
        self.vital_history['timestamps'].append(now)
        self.vital_history['heart_rate'].append(self.heart_rate)
        self.vital_history['spo2'].append(self.spo2)
        self.vital_history['temperature'].append(self.temperature)

        # 状态时间线
        self.state_timeline.append({
            'time': now,
            'state_id': self.state_id,
            'state_name': self.state_name,
            'alert_level': alert_level,
        })

        # ── 无人活动计时（白天累计，黑夜不计时）──
        self._update_inactive_timer()

        # 告警记录
        alert = self._check_and_generate_alert(alert_level)
        if alert:
            self.alert_history.append(alert)
            return alert
        return None

    def _update_inactive_timer(self):
        """
        更新无人活动计时器，只累计白天时段。

        当状态为"无人活动"(state_id=6)时，计算自上次检查以来的白昼秒数。
        当状态变为其他时，重置所有计时状态。
        """
        now = datetime.now()
        is_inactive = (self.state_id == 6)

        if is_inactive:
            # 记录首次无人活动的时刻
            if self.inactive_start is None:
                self.inactive_start = now

            # 计算自上次检查以来的白昼秒数
            if self.last_inactive_check is not None:
                daytime_sec = daytime_seconds_between(self.last_inactive_check, now)
                self.daytime_inactive_seconds += daytime_sec

            self.last_inactive_check = now
        else:
            # 有人活动 → 重置所有计时
            self.inactive_start = None
            self.daytime_inactive_seconds = 0.0
            self.last_inactive_check = None
            self._inactive_alert_sent = False

    def _check_and_generate_alert(self, alert_level: str):
        """
        检查是否需要生成告警。

        对于 state_id=6 (无人活动)：
        —— 需要白天累计满 config.INACTIVITY_ALERT['inactive_hours'] 小时才触发，
           且同一段无人活动中只触发一次。

        对于 state_id=4,5 及其他异常：
        —— 按原有逻辑立即触发。
        """
        if self.state_id == 6:
            # 无人活动告警：白天累计满 N 小时才触发
            threshold = config.INACTIVITY_ALERT["inactive_hours"] * 3600
            if self.daytime_inactive_seconds >= threshold and not self._inactive_alert_sent:
                self._inactive_alert_sent = True
                return {
                    'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'level': alert_level,
                    'message': self._alert_message(),
                }
            return None

        # 其他告警类型（跌倒、异常姿态、心率异常等）保持不变
        if self.state_id in (4, 5) or (self.heart_rate and (self.heart_rate < 40 or self.heart_rate > 120)):
            return {
                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'level': alert_level,
                'message': self._alert_message(),
            }
        return None

    def _alert_message(self):
        """生成告警消息文本"""
        if self.state_id == 5:
            return "🚨 跌倒告警！老人可能已摔倒，请立即确认！"
        elif self.state_id == 6:
            hours = self.daytime_inactive_seconds / 3600
            return f"⚠️ 失联告警！白天累计无人活动已达 {hours:.1f} 小时（黑夜不计时）"
        elif self.state_id == 4:
            return "🟠 异常姿态！老人姿势异常，建议关注"
        elif self.heart_rate and self.heart_rate > 120:
            return f"❤️ 心率异常偏高: {self.heart_rate} bpm"
        elif self.heart_rate and self.heart_rate < 40:
            return f"❤️ 心率异常偏低: {self.heart_rate} bpm"
        return "检查老人状态"

    def to_dict(self):
        """序列化为 JSON"""
        # 格式化无人活动时长
        if self.state_id == 6 and self.daytime_inactive_seconds > 0:
            h = int(self.daytime_inactive_seconds // 3600)
            m = int((self.daytime_inactive_seconds % 3600) // 60)
            inactive_str = f"{h}小时{m}分钟"
        else:
            inactive_str = None

        return {
            'device_id': self.device_id,
            'state_id': self.state_id,
            'state_name': self.state_name,
            'confidence': self.confidence,
            'heart_rate': self.heart_rate,
            'spo2': self.spo2,
            'temperature': self.temperature,
            'battery': self.battery,
            'last_update': self.last_update,
            'is_online': self.is_online,
            'alert_level': self.alert_level,
            'alert_color': self.alert_color,
            'alert_history': list(self.alert_history),
            'state_timeline': list(self.state_timeline),
            'vital_history': {
                'timestamps': list(self.vital_history['timestamps']),
                'heart_rate': list(self.vital_history['heart_rate']),
                'spo2': list(self.vital_history['spo2']),
                'temperature': list(self.vital_history['temperature']),
            },
            # 无人活动计时信息
            'inactive_info': {
                'is_inactive': self.state_id == 6,
                'daytime_seconds': self.daytime_inactive_seconds,
                'daytime_str': inactive_str,
                'started_at': self.inactive_start.isoformat() if self.inactive_start else None,
                'alert_threshold_hours': config.INACTIVITY_ALERT["inactive_hours"],
                'alert_sent': self._inactive_alert_sent,
                'daytime_range': f"{config.INACTIVITY_ALERT['daytime_start']:02d}:00-{config.INACTIVITY_ALERT['daytime_end']:02d}:00",
            },
        }


# 全局单例
system_state = SystemState()

# 页面访问计数
stats = {
    'start_time': time.time(),
    'total_visits': 0,
    'total_api_calls': 0,
}


# ============================================================
# 路由 — 页面
# ============================================================

@app.route('/')
def index():
    """主仪表盘页面"""
    stats['total_visits'] += 1
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """获取当前完整状态（供前端初始化用）"""
    return jsonify(system_state.to_dict())


@app.route('/api/stats')
def api_stats():
    """服务器统计信息"""
    uptime = time.time() - stats['start_time']
    return jsonify({
        'uptime_seconds': uptime,
        'uptime_str': f"{uptime/3600:.1f}h",
        'total_visits': stats['total_visits'],
        'total_api_calls': stats['total_api_calls'],
        'start_time': datetime.fromtimestamp(stats['start_time']).strftime('%Y-%m-%d %H:%M:%S'),
    })


# ============================================================
# 路由 — 开发板数据上传接口
# ============================================================

@app.route('/api/data', methods=['POST'])
def receive_data():
    """
    开发板上传数据的主接口
    支持的 JSON 格式见文件头部注释
    """
    stats['total_api_calls'] += 1

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({'error': '无效的 JSON 数据'}), 400

    # 必需字段检查
    if 'state_id' not in data and 'state_name' not in data:
        return jsonify({'error': '缺少 state_id 或 state_name 字段'}), 400

    # 更新系统状态
    alert = system_state.update_from_api(data)

    # WebSocket 广播更新
    socketio.emit('state_update', system_state.to_dict())

    # 如果有告警，单独推送
    if alert:
        socketio.emit('alert', alert)

    return jsonify({
        'status': 'ok',
        'alert': alert,
        'timestamp': int(time.time() * 1000),
    })


@app.route('/api/heartbeat', methods=['POST'])
def heartbeat():
    """开发板心跳接口（保持在线状态）"""
    system_state.is_online = True
    system_state.last_update = time.time()
    return jsonify({'status': 'ok'})


# ============================================================
# WebSocket 事件
# ============================================================

@socketio.on('connect')
def handle_connect():
    """浏览器连接时推送当前状态"""
    print(f"🔗 浏览器连接: {request.sid}")
    emit('state_update', system_state.to_dict())


@socketio.on('disconnect')
def handle_disconnect():
    print(f"🔌 浏览器断开: {request.sid}")


# ============================================================
# 视频流（可选）
# ============================================================

class VideoStream:
    """摄像头视频流（后台线程读取）"""

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame_queue = queue.Queue(maxsize=30)
        self.running = True
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def _capture(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # 编码为 JPEG
                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(jpeg.tobytes())
            else:
                time.sleep(0.1)

    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return None

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2)
        self.cap.release()


# 视频流实例（按需启动）
video_stream = None


def gen_mjpeg():
    """MJPEG 视频流生成器"""
    global video_stream
    if video_stream is None:
        video_stream = VideoStream(config.CAMERA_INDEX)

    while True:
        frame = video_stream.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            yield (b'--frame\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    """MJPEG 视频流端点"""
    return Response(
        gen_mjpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="护龄 Web 监控服务器")
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='绑定地址 (默认: 0.0.0.0 允许局域网访问)')
    parser.add_argument('--port', type=int, default=5000,
                        help='端口 (默认: 5000)')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式')
    parser.add_argument('--no-video', action='store_true',
                        help='禁用摄像头视频流')
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════╗
║                                                  ║
║   护龄 v3 — Web 监控服务器                        ║
║   基于 Flask + Socket.IO 实时数据推送             ║
║                                                  ║
╠══════════════════════════════════════════════════╣
║   仪表盘: http://localhost:{args.port}              ║
║   视频流: http://localhost:{args.port}/video_feed   ║
║   API:    POST http://localhost:{args.port}/api/data║
║                                                  ║
╠══════════════════════════════════════════════════╣
║   {len(config.STATE_NAMES)} 类状态标签:                              ║
║   0=正常行走  1=坐着/休息  2=躺卧  3=久坐未动  ║
║   4=异常姿态  5=跌倒/倒地  6=无人活动         ║
╚══════════════════════════════════════════════════╝
""")

    print(f"🚀 服务启动中...")
    print(f"   📡 绑定: {args.host}:{args.port}")
    print(f"   🎥 视频流: {'已启用' if not args.no_video else '已禁用'}")
    print(f"\n💡 开发板对接示例:")
    print(f'   curl -X POST http://<本机IP>:{args.port}/api/data \\')
    print(f'     -H "Content-Type: application/json" \\')
    print(f'     -d \'{{"device_id":"huling_001","state_id":1,"heart_rate":72}}\'')
    print()

    try:
        socketio.run(
            app,
            host=args.host,
            port=args.port,
            debug=args.debug,
            allow_unsafe_werkzeug=True,
        )
    except KeyboardInterrupt:
        print("\n⏹ 服务已停止")
    finally:
        global video_stream
        if video_stream:
            video_stream.release()


if __name__ == '__main__':
    main()
