# 香橙派 5 Pro 部署教程

> 硬件：Orange Pi 5 Pro（RK3588S，4GB LPDDR5）  
> 系统：Armbian / Orange Pi OS（aarch64）  
> 接续：ONNX 模型已导出 + int8 量化完成

---

## 〇、部署总览

```
PC（训练机）                                     香橙派 5 Pro
┌──────────────────┐          scp 传输         ┌──────────────────────┐
│ model.int8.onnx  │ ──────────────────────────→│  ~/asr/              │
│ tokens.txt       │                            │  ├── model.int8.onnx │
│                  │                            │  └── tokens.txt      │
└──────────────────┘                            │                      │
                                                │  安装 sherpa-onnx    │
                                                │  连接 USB 麦克风      │
                                                │  启动实时转写 🎙️      │
                                                └──────────────────────┘
```

---

## 一、香橙派系统准备

### 1.1 SSH 登录

```bash
# 在 PC 上
ssh orangepi@<香橙派IP地址>
```

> 默认用户名密码通常是 `orangepi` / `orangepi`  
> IP 地址可在路由器后台查看，或香橙派接显示器用 `ip addr` 查看

### 1.2 基础环境

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装必需品
sudo apt install -y \
    python3 python3-pip python3-venv \
    portaudio19-dev libportaudio2 \
    wget tar bzip2 \
    alsa-utils

# 查看 Python 版本（需要 3.8+）
python3 --version
```

### 1.3 创建虚拟环境

```bash
python3 -m venv ~/asr_env
source ~/asr_env/bin/activate

# 升级 pip
pip install --upgrade pip setuptools wheel

# 验证
which python
# → /home/orangepi/asr_env/bin/python
```

---

## 二、安装 sherpa-onnx

```bash
# 确保在虚拟环境中
source ~/asr_env/bin/activate

# 安装 sherpa-onnx（aarch64 自动适配）
pip install sherpa-onnx

# 验证安装
sherpa-onnx --help
# 出现帮助信息 = 安装成功 ✅

# 查看可用命令
ls $(dirname $(which sherpa-onnx)) | grep sherpa-onnx
```

可用二进制工具：

| 工具 | 用途 |
|------|------|
| `sherpa-onnx-offline` | 离线文件转写 |
| `sherpa-onnx-microphone-offline` | 麦克风 → 转写 |
| `sherpa-onnx-vad-microphone-offline-asr` | VAD + 麦克风 → 转写（**推荐**） |
| `sherpa-onnx-vad` | 纯语音活动检测 |

---

## 三、传输模型文件

### 3.1 从 PC 传到香橙派

```bash
# 在 PC 的 WSL / 终端中执行
scp ~/sensevoice_deploy/model.int8.onnx orangepi@<IP>:~/asr/
scp ~/sensevoice_deploy/tokens.txt orangepi@<IP>:~/asr/

# 或者在香橙派上拉取
# ssh orangepi@<IP>
# scp user@<PC_IP>:~/sensevoice_deploy/model.int8.onnx ~/asr/
```

### 3.2 Windows 直接传文件

用 **WinSCP** 或 **FileZilla** 等 SFTP 工具，连接到香橙派 IP，用户名 `orangepi`，把文件拖到 `/home/orangepi/asr/`。

### 3.3 U 盘拷贝

```bash
# 插上 U 盘后
sudo mount /dev/sda1 /mnt
cp /mnt/model.int8.onnx ~/asr/
cp /mnt/tokens.txt ~/asr/
sudo umount /mnt
```

---

## 四、连接麦克风

### 4.1 USB 麦克风（推荐）

插上 USB 麦克风后：

```bash
# 查看音频设备
arecord -l

# 典型输出：
# card 2: Device [USB Audio Device], device 0: USB Audio [USB Audio]
#   Subdevices: 1/1

# 测试录音（对着麦克风说话，Ctrl+C 停止）
arecord -f S16_LE -r 16000 -c 1 -d 5 test.wav
aplay test.wav  # 回放测试
```

### 4.2 设置默认设备（可选）

如果有多个音频设备：

```bash
# 查看所有录音设备
cat /proc/asound/cards

# 设置默认设备（把 card 编号替换为你的 USB 麦克风编号）
# 编辑 ~/.asoundrc
cat > ~/.asoundrc << 'EOF'
pcm.!default {
    type hw
    card 2          # 改为你的 USB 麦克风 card 编号
    device 0
}

ctl.!default {
    type hw
    card 2
}
EOF
```

### 4.3 调节麦克风音量

```bash
# 图形界面
alsamixer
# 按 F4 切换到 Capture 视图，上下箭头调音量

# 命令行
amixer sset 'Mic' 80%
```

---

## 五、运行实时转写

### 5.1 快速测试（文件转写）

先用一段测试音频验证模型能正常工作：

```bash
source ~/asr_env/bin/activate

# 录制 5 秒测试音频
arecord -f S16_LE -r 16000 -c 1 -d 5 ~/test.wav

# 转写
sherpa-onnx-offline \
  --tokens=~/asr/tokens.txt \
  --sense-voice-model=~/asr/model.int8.onnx \
  --num-threads=6 \
  --sense-voice-use-itn=1 \
  ~/test.wav
```

### 5.2 麦克风实时转写（基础版）

```bash
source ~/asr_env/bin/activate

sherpa-onnx-microphone-offline \
  --tokens=~/asr/tokens.txt \
  --sense-voice-model=~/asr/model.int8.onnx \
  --num-threads=6 \
  --sense-voice-use-itn=1
```

按 `Ctrl+C` 停止。

### 5.3 VAD + 实时转写（推荐版）

这个版本带语音活动检测，不说话时不占用 CPU：

```bash
source ~/asr_env/bin/activate

sherpa-onnx-vad-microphone-offline-asr \
  --tokens=~/asr/tokens.txt \
  --sense-voice-model=~/asr/model.int8.onnx \
  --num-threads=6 \
  --sense-voice-use-itn=1
```

> 📌 快速说话 → 停顿 → 自动输出结果 → 继续聆听

---

## 六、启动脚本

### 6.1 一键运行脚本

创建 `~/asr/run.sh`：

```bash
#!/bin/bash
# run.sh - 启动实时中文语音转写
set -e

echo "🎙️  中文实时语音转写"
echo "===================="

# 激活环境
source ~/asr_env/bin/activate

# 检查模型
if [ ! -f ~/asr/model.int8.onnx ]; then
    echo "❌ 模型文件不存在: ~/asr/model.int8.onnx"
    echo "请先用 scp 把模型传到香橙派"
    exit 1
fi

if [ ! -f ~/asr/tokens.txt ]; then
    echo "❌ tokens.txt 不存在"
    exit 1
fi

# 检查麦克风
if ! arecord -l 2>/dev/null | grep -q card; then
    echo "❌ 未检测到麦克风，请检查连接"
    exit 1
fi

echo "✅ 模型已就绪"
echo "✅ 麦克风已连接"
echo ""
echo "开始监听...（Ctrl+C 停止）"
echo ""

# 启动转写
sherpa-onnx-vad-microphone-offline-asr \
  --tokens=~/asr/tokens.txt \
  --sense-voice-model=~/asr/model.int8.onnx \
  --num-threads=6 \
  --sense-voice-use-itn=1
```

```bash
chmod +x ~/asr/run.sh
```

### 6.2 开机自启（可选）

```bash
# 创建 systemd 服务
sudo tee /etc/systemd/system/asr.service << 'EOF'
[Unit]
Description=Real-time Chinese ASR
After=network.target sound.target

[Service]
Type=simple
User=orangepi
WorkingDirectory=/home/orangepi/asr
ExecStart=/home/orangepi/asr/run.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 启用自启
sudo systemctl enable asr.service
sudo systemctl start asr.service

# 查看状态
sudo systemctl status asr.service

# 查看实时日志
sudo journalctl -u asr.service -f
```

---

## 七、性能调优

### 7.1 线程数调优

```bash
# RK3588S 是 8 核 CPU
# 建议 --num-threads 设为 4~6，留核心给系统和音频 I/O
# 实测哪个最快用哪个：

for t in 4 5 6 7; do
    echo "--- 测试 $t 线程 ---"
    time sherpa-onnx-offline \
        --tokens=~/asr/tokens.txt \
        --sense-voice-model=~/asr/model.int8.onnx \
        --num-threads=$t \
        ~/test.wav 2>&1 | grep "Elapsed"
done
```

### 7.2 CPU 频率锁定

```bash
# 查看当前频率
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq

# 锁定性能模式
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 恢复省电模式
# echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 7.3 内存清理

```bash
# 转写前清理缓存，确保有足够内存
sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
free -h
```

---

## 八、NPU 加速（进阶可选）

> ⚠️ 需要 RKNN 版 sherpa-onnx + RKNN 格式模型

### 8.1 检查 NPU 驱动

```bash
# 检查 librknnrt.so
strings /lib/librknnrt.so 2>/dev/null | grep "librknnrt version"
# 期望：librknnrt version: 2.1.0 或 2.2.0

# 如果没有，说明当前系统镜像不含 NPU 驱动
# 需要刷入 Orange Pi 官方镜像或更新内核
```

### 8.2 安装 RKNN 版 sherpa-onnx

```bash
# 先卸载普通版
pip uninstall sherpa-onnx -y

# 安装 RKNN 专用版
pip install sherpa-onnx -f https://k2-fsa.github.io/sherpa/onnx/rk-npu.html

# 验证
ldd $(which sherpa-onnx) | grep rknn
# 必须看到 librknnrt.so
```

### 8.3 转换 ONNX → RKNN（在 PC 上做）

RKNN 转换需要 x86_64 Linux（WSL2 可以），装 `rknn-toolkit2`：

```bash
# 在 PC 的 WSL2 中
pip install rknn-toolkit2
```

转换脚本 `convert_to_rknn.py`：

```python
from rknn.api import RKNN

rknn = RKNN(verbose=True)

# 加载 ONNX
ret = rknn.load_onnx(
    model='./model.int8.onnx',
)
print(f"load_onnx: {ret}")

# 配置（RK3588）
ret = rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rk3588',
)
print(f"config: {ret}")

# 构建
ret = rknn.build(do_quantization=False)
print(f"build: {ret}")

# 导出
ret = rknn.export_rknn('./model.rknn')
print(f"export: {ret}")

rknn.release()
print("✅ RKNN 模型导出完成")
```

### 8.4 NPU 推理

```bash
# 传输 model.rknn 到香橙派
scp model.rknn orangepi@<IP>:~/asr/

# 在香橙派上运行
sherpa-onnx-vad-microphone-offline-asr \
  --provider=rknn \
  --tokens=~/asr/tokens.txt \
  --sense-voice-model=~/asr/model.rknn
```

---

## 九、故障排查

| 问题 | 解决方案 |
|------|----------|
| `ModuleNotFoundError: sherpa_onnx` | `source ~/asr_env/bin/activate` |
| 麦克风没声音 | `arecord -l` 查看设备，`alsamixer` 调音量 |
| 转写全是错字 | 检查音频采样率是否为 16kHz |
| CPU 100% 一直转 | 加上 VAD（`sherpa-onnx-vad-microphone-offline-asr`） |
| 内存不足 (OOM) | 确认用 int8 模型（~228MB），检查 `free -h` |
| 延迟太高 | 减少 `--num-threads`、检查散热、锁定 CPU 频率 |
| `librknnrt.so not found` | 系统不含 NPU 驱动，升级系统镜像 |
| SSH 连接断开 | `sudo systemctl restart sshd`，或改用串口 |

---

## 十、性能参考

| 模式 | RTF（实时率） | CPU 占用 | 延迟 |
|------|:---:|:---:|:---:|
| CPU (int8, 6线程) | 0.3-0.5 | 60-80% | 1-2s |
| CPU (int8, 4线程) | 0.4-0.6 | 40-60% | 1-3s |
| NPU (RKNN) | 0.05-0.1 | <10% | <0.5s |

> **RTF 0.3 的含义**：1 秒音频需要 0.3 秒处理，完全跟得上实时。

---

## 附录：目录结构总览

```
/home/orangepi/asr/
├── run.sh                # 一键启动脚本
├── model.int8.onnx       # 微调后的 ONNX int8 模型（~228MB）
├── tokens.txt            # 词表
├── test.wav              # 测试音频
│
├── scripts/              # 工具脚本
│   ├── record.sh         # 录音脚本
│   ├── monitor.sh        # 资源监控脚本
│   └── transcribe.sh     # 批量转写脚本
│
└── logs/                 # 日志
    └── asr_$(date).log
```

---

> 📅 最后更新：2026-07-16  
> 🍊 硬件：Orange Pi 5 Pro / RK3588S / 4GB LPDDR5  
> 📦 推理引擎：sherpa-onnx  
> 🎯 模型：SenseVoiceSmall (int8)
