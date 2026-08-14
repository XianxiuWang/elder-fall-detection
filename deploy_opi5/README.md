# 老人跌倒检测 — Orange Pi 5 Pro 部署包

> 模型: XGBClassifier 6 分类 (V5+新数据, 98.71%)  
> 管线: RGB 视频 → MediaPipe Pose → 30帧滑窗 → 分类器 → 时序过滤器 v3  
> 平台: Orange Pi 5 Pro (RK3588S ARM64)

## 文件清单

```
deploy_opi5/
├── fall_inference.py              # ★ 自包含推理脚本（核心）
├── install.sh                     # 一键安装脚本
├── requirements_opi5.txt          # ARM64 Python 依赖
├── README.md                      # 本文档
└── models/
    └── fall_classifier_6class.pkl  # V5 模型 (~3MB)
```

## 快速开始

### 1. 把文件传到板子上

```bash
# 从 PC 传到 Orange Pi（用 scp 或 U 盘）
scp -r deploy_opi5/ orangepi@192.168.x.x:~/fall_detection/
```

### 2. 安装

```bash
cd ~/fall_detection
bash install.sh
```

### 3. 测试

```bash
# 激活环境
source venv/bin/activate

# 跑一段视频
python3 fall_inference.py test_video.mp4

# 性能基准测试（不依赖摄像头）
python3 fall_inference.py --benchmark

# 实时摄像头
python3 fall_inference.py --realtime
```

## 接入萤石 RTSP 摄像头（到货即用）

`cv_client.py` 已内置 RTSP 支持，摄像头到货后开箱即用：

```bash
# 方式 1：直接指定完整 RTSP 地址
python3 cv_client.py --rtsp "rtsp://admin:验证码@192.168.1.100:554/h264/ch1/sub/av_stream"

# 方式 2：只填 IP + 设备验证码，自动拼接（推荐）
python3 cv_client.py --ezviz-ip 192.168.1.100 --ezviz-code ABCDEF

# 子码流（默认，推理流畅）/ 主码流（高清）
python3 cv_client.py --ezviz-ip 192.168.1.100 --ezviz-code ABCDEF --ezviz-stream main
```

> **注意**：需在「萤石云视频」App 中开启 RTSP：设备设置 → 本地服务设置 → 开启 RTSP。
> 设备验证码是摄像头底部标签上的 6 位大写字母。断流后会自动重连。

## 用法说明

```
python3 fall_inference.py <video.mp4> [选项]

选项:
  --save-video        输出标注视频（带 Fall 红框）
  --model PATH        指定模型路径
  --complexity {0,1,2} MediaPipe 复杂度（0=最快, 1=平衡, 2=最准）
  --realtime          实时摄像头模式
  --camera ID         摄像头 ID（默认 0）
  --benchmark         性能基准测试
```

## 输出格式

运行后生成 `results_<视频名>.json`：

```json
{
  "video": "test.mp4",
  "duration_s": 120.0,
  "config": { "window_size": 30, "stride": 6, ... },
  "stats": {
    "n_predictions": 400,
    "filt_falls": 15,
    "fall_events": [{"start": 45.2, "end": 47.8}],
    "processing_fps": 8.5
  },
  "predictions": [
    {"frame": 0, "time_s": 0.0, "raw": "Standing", "filt": "Standing", "fall_prob": 0.02},
    ...
  ]
}
```

## 性能调优

Orange Pi 5 Pro (RK3588S) 的瓶颈在 **MediaPipe Pose**（关键点提取），不在分类器。

| 参数 | 推荐值 | 说明 |
|:---|:---|:---|
| MODEL_COMPLEXITY | 1 | 平衡速度与精度 |
| SKIP_FRAMES | 3 | 每 3 帧取 1 帧，30fps → 10fps 有效 |
| WINDOW_SIZE | 30 | 约 3 秒上下文（10fps 有效帧率下） |
| POSE_CHUNK | 200 | 每 200 帧重建一次 MediaPipe |

**如果速度不够：**
```bash
# 降低复杂度（可能在边缘姿势上准确性降低）
python3 fall_inference.py video.mp4 --complexity 0

# 增大跳帧（减少 CPU 负载）
# 编辑 fall_inference.py: SKIP_FRAMES = 5
```

**预期性能（Orange Pi 5 Pro）：**
- MediaPipe complexity=1: ~5-10 fps（有效帧率）
- MediaPipe complexity=0: ~12-18 fps（有效帧率）
- XGBoost 推理: ~0.5ms/window（几乎无开销）

## NPU 加速（可选，高级）

RK3588S 有 6 TOPS NPU，可用于加速 MediaPipe Pose：
- 方案：使用 RKNN 转换的 Pose Landmark 模型
- 参考：https://github.com/rockchip-linux/rknn-toolkit2
- 注意：需要自行编译 RKNN 版本的 MediaPipe，不在本部署包范围内

## 排查

### MediaPipe 安装失败
```bash
# 手动尝试
pip install mediapipe==0.10.14
# 或从 piwheels
pip install mediapipe --index-url https://www.piwheels.org/simple
```

### OpenCV 无法打开摄像头
```bash
# 检查摄像头
v4l2-ctl --list-devices
# 尝试不同 ID
python3 fall_inference.py --realtime --camera 1
```

### 内存不足
```bash
# 减小 POSE_CHUNK（编辑 fall_inference.py）
POSE_CHUNK = 100  # 从 200 降到 100
```
