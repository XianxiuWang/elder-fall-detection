# Orange Pi 5 Pro NPU 环境搭建

> 项目：基于多模态 AI 监测的老年人跌倒风险研究
> 硬件：Orange Pi 5 Pro (RK3588S, 6 TOPS NPU)

---

## 文件说明

| 文件 | 运行位置 | 作用 |
|------|---------|------|
| `verify_npu.py` | 板子端 | 一键检测 NPU 环境是否就绪 |
| `convert.py` | PC 端 | ONNX 模型 → RKNN 格式转换 |
| `infer.py` | 板子端 | 加载 RKNN 模型，NPU 推理基准测试 |

---

## 操作流程

### Step 1: 板子刷系统

```
1. 下载 Orange Pi OS (Arch) 或 Ubuntu 22.04 镜像 (orangepi.org)
2. balenaEtcher 烧录到 SD 卡 (≥32GB)
3. 插卡 → 接显示器/键鼠 → Type-C 电源 (5V/4A) → 开机
4. 联网: sudo apt update && sudo apt upgrade -y
5. 装依赖: sudo apt install -y python3-pip git
```

### Step 2: 板子端验证环境

```bash
# 把 verify_npu.py 传到板子上
scp verify_npu.py orangepi@<IP>:/home/orangepi/

# SSH 登录板子，运行检测
ssh orangepi@<IP>
python3 verify_npu.py
```

应该看到全部 ✅。如果没有 NPU 驱动，需要换官方镜像。

### Step 3: PC 端安装 RKNN Toolkit2

```bash
git clone https://github.com/airockchip/rknn-toolkit2.git
pip install rknn-toolkit2/packages/rknn_toolkit2-*-cp310-*.whl

# 验证
python -c "from rknn.api import RKNN; print('OK')"
```

### Step 4: 模型转换（PC 端）

```bash
# 以 MediaPipe Pose 的 ONNX 模型为例
python convert.py --model pose_landmarker.onnx --output pose_landmarker.rknn
```

### Step 5: 推理验证（板子端）

```bash
# 把 .rknn 模型 + infer.py 传到板子上
scp infer.py pose_landmarker.rknn orangepi@<IP>:/home/orangepi/

# 板子上跑推理
ssh orangepi@<IP>
python3 infer.py --model pose_landmarker.rknn --loop 20
```

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 板子不亮灯 | 电源不够 | 必须 5V/4A Type-C，手机充电器不行 |
| `rknn.init_runtime()` 失败 | NPU 驱动未加载 | 换 Orange Pi 官方镜像 |
| `dmesg` 没有 rknpu | 镜像不含 NPU 驱动 | 下载带 "RK3588" 字样的镜像 |
| 推理很慢 (> 50ms/帧) | 模型未量化 | 用 `convert.py` 开 `do_quantization=True` |
| 转换报错"op not supported" | ONNX 用了不支持的算子 | 尝试 opset=12 导出，或简化模型结构 |
| 多次推理后越来越慢 | NPU 过热降频 | 加散热片+风扇 |

---

## 数据流图

```
┌─────────────┐     ONNX      ┌─────────────┐     RKNN     ┌─────────────────┐
│  PyTorch    │ ────────────→ │  PC 端      │ ───────────→ │  Orange Pi 5 Pro│
│  训练模型    │               │  convert.py │              │  6 TOPS NPU     │
│  (MediaPipe │               │  RKNN转工具  │              │  infer.py      │
│   YOLO等)   │               └─────────────┘              └─────────────────┘
└─────────────┘
```
