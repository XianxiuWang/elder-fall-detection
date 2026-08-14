# Orange Pi 5 Pro 初始化完整教程

> 目标：从空白板子到能跑 `fall_inference.py`  
> 硬件：Orange Pi 5 Pro (RK3588S, 4/8/16GB RAM)  
> 预计耗时：30-60 分钟（主要看下载速度）

---

## 准备清单

| 物品 | 说明 |
|:---|:---|
| Orange Pi 5 Pro 板子 | 本体 |
| MicroSD 卡 | 建议 32GB+，Class 10 / A1 |
| 读卡器 | 刷系统用 |
| 5V/4A Type-C 电源 | **必须足功率**，普通手机充电头可能不稳 |
| 网线（可选） | 首次配置更稳定，也可以纯 WiFi |
| HDMI 线 + 显示器（可选） | 首次配置方便，也可以纯 SSH 无头启动 |
| 键盘鼠标（可选） | 仅首次 HDMI 配置需要 |
| 散热片/风扇 | **必须**，RK3588S 跑 MediaPipe 会严重发热 |

---

## 第一步：下载系统镜像

推荐 **Armbian**（社区维护好，驱动完善）或 **Orange Pi 官方 Ubuntu**。

### 方案 A：Armbian（推荐）

1. 打开 https://www.armbian.com/orangepi-5-pro/
2. 下载最新 **Ubuntu Noble (24.04) Server** 或 **Desktop** 版
   - Server 版：无桌面，省资源，适合纯推理
   - Desktop 版：有桌面，方便调试（需要接显示器）
3. 推荐选 Server 版 + SSH 远程管理

### 方案 B：Orange Pi 官方

1. 打开 http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-5-Pro.html
2. 下载 **Orangepi5pro_1.0.4_ubuntu_jammy_desktop_xfce_linux6.1.43** 或更新版本
3. 官方镜像稳定性可能不如 Armbian，但 NPU 驱动配套更好

---

## 第二步：烧录系统到 SD 卡

### Windows 上（你现在用的）

1. 下载 **balenaEtcher**：https://www.balena.io/etcher/
2. 或用 **Rufus**：https://rufus.ie/
3. 插入 SD 卡 → 打开 Etcher → 选镜像 → 选 SD 卡 → Flash
4. 烧完**不要拔卡**，继续下一步配置

### 首次启动前配置（重要！）

烧完后 SD 卡会显示为 `boot` 分区（约 256MB FAT32），在这里做预配置：

#### 3.1 启用 SSH（Armbian）

在 `boot` 分区根目录创建一个空文件：
```
boot/
└── ssh          ← 新建这个空文件，无后缀名
```
Armbian 开机后检测到这个文件会自动启用 SSH。

#### 3.2 预配置 WiFi（无头启动用）

在 `boot` 分区编辑 `armbian_first_run.txt`（Armbian）：
```
FR_general_delete_this_file_after_completion=1
FR_net_change_defaults=1
FR_net_ethernet_enabled=1
FR_net_wifi_enabled=1
FR_net_wifi_ssid='你的WiFi名'
FR_net_wifi_key='你的WiFi密码'
FR_net_wifi_countrycode='CN'
FR_net_use_static=0
```

如果是 Orange Pi 官方镜像，WiFi 需要在开机后用 `nmtui` 配置。

---

## 第三步：首次开机

1. SD 卡插入 Orange Pi 5 Pro
2. **接上散热风扇**（必须！）
3. 插网线（推荐）或靠 WiFi 预配置
4. 插 Type-C 电源 → 红灯亮，绿灯闪烁 = 正常启动
5. 等 1-2 分钟（首次启动慢，要扩展分区）

### 获取 IP 地址

**方法 1：路由器后台**
- 登录路由器管理页（通常是 192.168.1.1）
- 查看 DHCP 客户端列表，找到 `orangepi5-pro` 或 `armbian`

**方法 2：IP 扫描（Windows）**
```cmd
arp -a | findstr /i "orangepi armbian"
```

**方法 3：接显示器**
- 插 HDMI → 开机 → 登录后输入 `ip a`

---

## 第四步：SSH 登录 + 基础配置

```bash
# 从你的 PC SSH 进去
ssh root@192.168.x.x
# 或
ssh orangepi@192.168.x.x

# Armbian 初始密码: 1234（首次登录会强制改密码）
# Orange Pi 官方默认: orangepi / orangepi
```

首次登录 Armbian 会引导你做：
1. 改 root 密码
2. 创建普通用户（建议建一个：`opi`）
3. 选时区（Asia/Shanghai）
4. 选语言（en_US.UTF-8）

### 基础优化

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y curl wget git vim htop i2c-tools net-tools

# 设置时区
sudo timedatectl set-timezone Asia/Shanghai

# 固定 IP（如果用网线且不想每次都查 IP）
# 编辑 /etc/netplan/ 或通过 nmtui 配置
```

---

## 第五步：安装 Python 环境

```bash
# Armbian/Ubuntu 自带 Python 3.10+，检查一下
python3 --version  # 应该 ≥ 3.10

# 安装 pip 和 venv
sudo apt install -y python3-pip python3-venv python3-dev

# 安装系统级依赖（MediaPipe 需要）
sudo apt install -y \
    libopencv-dev \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    libgomp1 \
    ffmpeg
```

---

## 第六步：传输部署包

从你的 Windows PC 传到 Orange Pi：

```cmd
# 方法 1：scp（推荐）
scp -r E:\老人跌倒\deploy_opi5 opi@192.168.x.x:~/fall_detection/

# 方法 2：U 盘
# 1. 把 deploy_opi5/ 文件夹拷到 U 盘
# 2. U 盘插 Orange Pi
# 3. 在 Orange Pi 上:
sudo mount /dev/sda1 /mnt
cp -r /mnt/deploy_opi5 ~/fall_detection/
```

---

## 第七步：安装 Python 依赖

```bash
cd ~/fall_detection
bash install.sh
```

**如果 MediaPipe 安装失败**（ARM64 最常见的卡点）：

```bash
# 尝试 1：指定版本
pip install mediapipe==0.10.14

# 尝试 2：从 piwheels 预编译源
pip install mediapipe --index-url https://www.piwheels.org/simple

# 尝试 3：如果上面都失败，检查 Python 版本
python3 --version
# MediaPipe 0.10.14 需要 Python 3.8-3.11
# 如果 Python 3.12+，需要 mediapipe>=0.10.18
pip install mediapipe==0.10.18
```

---

## 第八步：验证部署

```bash
source venv/bin/activate

# 1. 先检查所有依赖
python3 -c "
import numpy; print('numpy:', numpy.__version__)
import sklearn; print('sklearn:', sklearn.__version__)
import xgboost; print('xgboost:', xgboost.__version__)
import cv2; print('opencv:', cv2.__version__)
import mediapipe; print('mediapipe:', mediapipe.__version__)
print('All OK!')
"

# 2. 模型加载测试
python3 -c "
import pickle, os
p = os.path.join('models','fall_classifier_6class.pkl')
m = pickle.load(open(p,'rb'))
print('Model:', type(m['model']).__name__,
      m['feature_dim'], 'features',
      len(m['model'].classes_), 'classes')
print('Model OK!')
"

# 3. 性能基准测试（关键！看实际帧率）
python3 fall_inference.py --benchmark

# 4. 如果有测试视频，跑完整推理
python3 fall_inference.py test.mp4
```

---

## 散热警告 ⚠️

**RK3588S 跑 MediaPipe Pose 是满负载运算**，会迅速升温到 80°C+。

```bash
# 监控温度
watch -n 1 "cat /sys/class/thermal/thermal_zone0/temp | awk '{print \$1/1000\"°C\"}'"
```

**必须措施：**
- 安装散热片 + 风扇（5V PWM 风扇，板子上有风扇接口）
- Armbian 下配置风扇自动调速：

```bash
# 安装风扇控制
sudo apt install -y fancontrol
# 或手动设置
echo 255 | sudo tee /sys/class/pwm/pwmchip0/pwm0/duty_cycle
```

如果温度持续 >85°C，降低负载：设 `MODEL_COMPLEXITY=0`，增大 `SKIP_FRAMES=5`。

---

## 常见问题

### Q: 开机红灯亮但 SSH 连不上？
- 等 2 分钟（首次开机在扩展分区）
- 检查 SD 卡 `boot` 分区是否有 `ssh` 空文件
- 接 HDMI 显示器看启动日志

### Q: MediaPipe 报 `ImportError: libGL.so.1`？
```bash
sudo apt install -y libgl1-mesa-glx
```

### Q: 推理速度只有 2-3 fps？
- 正常，MediaPipe complexity=1 在 ARM 上就是这个速度
- 调优：`--complexity 0` + 增大 `SKIP_FRAMES`
- 如果想更快，需要把 Pose 模型转到 RKNN（NPU 加速，另外的项目）

### Q: 内存不够？
- 减少 `POSE_CHUNK` 到 100
- 关闭其他服务：`sudo systemctl disable bluetooth NetworkManager-wait-online`

---

## 下一步

初始化完成后：
1. 跑基准测试，记录实际 fps
2. 接 USB 摄像头或 RTSP 网络摄像头，跑 `--realtime`
3. 如需 NPU 加速，研究 RKNN Toolkit 2（https://github.com/rockchip-linux/rknn-toolkit2）
