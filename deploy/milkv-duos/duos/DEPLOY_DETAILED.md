# HuLing → Milk-V Duo S 部署完全指南

> **前提**：你需要一台 Windows PC（当前电脑）、Milk-V Duo S 开发板、microSD 卡（8GB+）、Type-C 数据线。

---

## 阶段 1：PC 端验证 C 代码 — 确保模型转换正确

### 1.1 确认你的环境

打开 **命令提示符 (cmd)** 或 **PowerShell**，执行：

```cmd
where gcc
```

如果显示 `gcc` 路径（如 `C:\msys64\mingw64\bin\gcc.exe`），说明已安装 MinGW，跳到 1.3。  
如果显示"找不到"，需要安装 MinGW。

### 1.2 安装 MinGW-w64（如果没有 gcc）

**方法A — 使用 MSYS2（推荐）：**
```
① 下载 https://github.com/msys2/msys2-installer/releases/download/2024-05-07/msys2-x86_64-20240507.exe
② 安装到 C:\msys64
③ 安装完成后自动打开 MSYS2 终端，执行：
   pacman -Syu
   pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-make
④ 将 C:\msys64\mingw64\bin 添加到系统 PATH
⑤ 重新打开 cmd，验证：gcc --version
```

### 1.3 检查 Python 环境

```cmd
python --version
# 需要 Python 3.8+

pip list | findstr mediapipe
pip list | findstr joblib
pip list | findstr m2cgen
```

如果缺少依赖：
```cmd
pip install mediapipe opencv-python joblib m2cgen scikit-learn numpy
```

### 1.4 生成测试数据（如果 test_data.h 不存在）

```cmd
cd D:\Users\wangxianxiu\.openclaw\workspace\huling_model
python generate_test_data.py
```

> 这个脚本会生成 `deploy/test_data.h`，包含 5 种姿态（站立/坐着/躺/跌倒/无人）的参考关键点和 Python 推理结果。

### 1.5 编译 C 验证程序

```cmd
cd D:\Users\wangxianxiu\.openclaw\workspace\huling_model\deploy

gcc -o test_deploy.exe huling_features.c random_forest.c random_forest_wrapper.c test_deploy.c -lm -O0 -Wall
```

**可能遇到的情况：**

| 错误信息 | 解决方法 |
|---------|---------|
| `fatal error: test_data.h: No such file or directory` | 回到第1.4步运行 `python generate_test_data.py` |
| `undefined reference to ...` | 确保加了 `-lm` 参数，且 `.c` 文件都在 |
| `random_forest.c: too large` 或编译很慢 | 这是正常的，文件有 84000 行，等 2-3 分钟 |

### 1.6 运行验证

```cmd
test_deploy.exe
```

**正确的输出：**
```
============================================================
  HuLing C Code Verification
============================================================
  Test cases: 6
  Features: 98
  Model classes: 3 (indices 0,1,5)

--- Case 0: standing ---
  torso: OK (6/6 match)
  joints: OK (66/66 match)
  angles: OK (8/8 match)
  structure: OK (8/8 match)
  motion: OK (4/4 match, all zeros)
  sensor: OK (6/6 match, all zeros)
...
============================================================
  ALL TESTS PASSED
============================================================
```

如果看到 `MISMATCH`，说明 C 代码与 Python 计算结果不一致，需要检查 `scaler_params.h` 和 `huling_features.c`。

---

## 阶段 2：获取 Milk-V Duo S SDK 和交叉编译工具链

> **这一步需要 Linux 环境**。推荐在 Windows 上安装 WSL2，或者用一台 Linux 机器/虚拟机。

### 2.1 安装 WSL2（如果还没有）

**以管理员身份打开 PowerShell，执行：**
```powershell
wsl --install
# 安装完成后重启电脑
# 重启后会自动打开 Ubuntu，设置用户名和密码
```

### 2.2 在 Ubuntu 中安装编译依赖

打开 Ubuntu 终端：
```bash
sudo apt update
sudo apt install -y build-essential cmake git wget cpio unzip rsync bc \
    libncurses-dev python3 python3-pip flex bison libssl-dev

# 验证
gcc --version   # 应显示版本号
git --version   # 应显示版本号
```

### 2.3 克隆 Buildroot SDK

```bash
cd ~
git clone https://github.com/milkv-duo/duo-buildroot-sdk.git
cd duo-buildroot-sdk
```

> ⚠️ 如果 `git clone` 很慢（国内网络），使用代理或 Gitee 镜像。

### 2.4 首次构建（生成工具链 + 系统镜像）

```bash
# 这一步需要 30-60 分钟，取决于电脑性能
./build.sh milkv-duos-sd
```

构建过程中的关键阶段：
1. **下载源码包**（~15分钟）— 从网上下载 Linux 内核、工具链等
2. **编译工具链**（~10分钟）— 生成 RISC-V/ARM 交叉编译器
3. **编译系统**（~20分钟）— 编译 Buildroot 根文件系统
4. **生成镜像**（~5分钟）— 打包为 SD 卡镜像

**如果构建中断（网络问题）：**
```bash
# 重新运行即可，Buildroot 会从断点继续
./build.sh milkv-duos-sd
```

### 2.5 验证工具链

```bash
# 查看产物
ls out/milkv-duos-sd/images/milkv-duos-sd.img
# 应看到该文件（系统镜像）

# 验证 RISC-V 编译器
ls host-tools/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-gcc
host-tools/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-gcc --version
# 输出: riscv64-unknown-linux-musl-gcc (version...) 

# 验证 ARM 编译器
ls host-tools/gcc/arm-buildroot-linux-gnueabihf/bin/arm-buildroot-linux-gnueabihf-gcc
```

### 2.6 记录 SDK 路径

```bash
echo "export MILKV_SDK=$HOME/duo-buildroot-sdk" >> ~/.bashrc
source ~/.bashrc
echo $MILKV_SDK
# 输出: /home/你的用户名/duo-buildroot-sdk
```

---

## 阶段 3：烧录系统并启动 Duo S

### 3.1 烧录镜像到 SD 卡

**在 Windows 上（推荐 Rufus）：**

```
① 下载 Rufus: https://rufus.ie
② 插入 microSD 卡到电脑
③ 打开 Rufus
   - 设备: 选择你的 SD 卡
   - 引导类型: 选择镜像文件
   - 点击"选择" → 找到 out/milkv-duos-sd/images/milkv-duos-sd.img
     （WSL 文件路径: \\wsl$\Ubuntu\home\你的用户名\duo-buildroot-sdk\out\...）
   - 点击"开始"
④ 等待烧录完成（约 2 分钟）
```

**在 Linux 上（dd 命令）：**
```bash
# 找到 SD 卡设备名
lsblk
# 看到类似 /dev/sdb (8GB) 的设备

# 烧录（⚠️ 确认设备名正确，选错会覆盖硬盘！）
sudo dd if=out/milkv-duos-sd/images/milkv-duos-sd.img of=/dev/sdb bs=4M status=progress
sync
```

### 3.2 启动 Duo S

```
① 将 microSD 卡插入 Duo S 背面的卡槽（金属触点朝上，Duo S 字面朝下）
② 用 Type-C 数据线连接 Duo S 的 Type-C 口（标注 "USB"）到电脑 USB 口
③ Duo S 上电后：
   - 蓝色 LED 常亮 = 已上电
   - 约 15 秒后 LED 闪烁 = 系统启动完成
```

### 3.3 安装 RNDIS 驱动（Windows）

首次连接时，Windows 会识别为 "RNDIS" 网络设备：

```
① 打开"设备管理器" → 看到一个带黄色感叹号的 "RNDIS" 设备
② 右键 → 更新驱动 → 浏览我的电脑
③ 驱动路径选择：C:\Windows\System32\DriverStore\FileRepository
④ 或下载 Milk-V 提供的 RNDIS 驱动：
   https://github.com/milkv-duo/duo-files/raw/main/duo-s/drivers/RNDIS.zip
⑤ 安装后，设备管理器中出现 "USB Ethernet/RNDIS Gadget" 网络适配器
```

### 3.4 验证网络连接

```cmd
# 打开 cmd
ping 192.168.42.1
```

如果 ping 通：
```
正在 Ping 192.168.42.1 具有 32 字节的数据:
来自 192.168.42.1 的回复: 字节=32 时间<1ms TTL=64
```

### 3.5 SSH 登录

```cmd
ssh root@192.168.42.1
# 密码: milkv
```

**首次登录显示：**
```
[root@milkv-duo]~# uname -a
Linux milkv-duo 5.10.4-2024.12.4 #1 PREEMPT ... riscv64 GNU/Linux
```

如果 SSH 提示 `connection refused`：
- 等 30 秒让系统完全启动
- 检查 RNDIS 驱动是否正确安装
- 或通过网络设置查看 IP 是否变了

---

## 阶段 4：交叉编译模型 C 代码

> 在 WSL/Linux Ubuntu 中执行。

### 4.1 进入 duos 目录

```bash
# 从 WSL 访问 Windows 文件系统
cd /mnt/d/Users/wangxianxiu/.openclaw/workspace/huling_model/deploy/duos

# 确认文件都在
ls -la
# 看到: CMakeLists.txt  main.c  toolchain-riscv.cmake  toolchain-arm.cmake
#       udp_server.c  test_deploy_duos.c  keypoint_bridge.py  DEPLOY.md
```

### 4.2 创建构建目录并编译

```bash
# 创建 RISC-V 构建目录
mkdir build-riscv
cd build-riscv

# CMake 配置（使用 RISC-V 工具链）
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv.cmake

# 如果报错 "CMAKE_C_COMPILER not found"，说明工具链路径不对
# 修改 toolchain-riscv.cmake 中的 SDK_ROOT 路径

# 编译
make -j$(nproc)
```

### 4.3 验证产物

```bash
ls -la huling_demo huling_test huling_server
file huling_demo
# 输出: huling_demo: ELF 64-bit LSB executable, UCB RISC-V, ...
```

**四个产物：**
| 文件 | 大小 | 用途 |
|------|------|------|
| `libhuling.a` | ~200KB | 静态库，可链接到其他项目 |
| `huling_demo` | ~250KB | 从 stdin/文件读取关键点推理 |
| `huling_test` | ~300KB | 对比 Python 参考值自检 |
| `huling_server` | ~260KB | UDP 实时推理服务器 |

### 4.4 可能遇到的问题

| 问题 | 解决方法 |
|------|---------|
| `cmake: command not found` | `sudo apt install cmake` |
| `The C compiler is not able to compile` | 检查 `toolchain-riscv.cmake` 中 `CROSS_PREFIX` 路径 |
| `random_forest.c: file too large` | 正常，等几分钟 |
| Out of memory | `make -j1` 单线程编译，减少内存占用 |

---

## 阶段 5：传输文件到 Duo S

### 5.1 通过 SCP 传输（推荐）

```bash
# 从 WSL/Ubuntu 传输到 Duo S
cd /mnt/d/Users/wangxianxiu/.openclaw/workspace/huling_model/deploy/duos/build-riscv

scp huling_demo huling_test huling_server root@192.168.42.1:/root/
# 密码: milkv
```

### 5.2 或者放到 SD 卡 FAT 分区

```bash
# Duo S 的 SD 卡有一个 FAT 分区可以被 Windows 识别
# 插入 SD 卡到电脑
cp huling_demo huling_test huling_server /mnt/sdcard/
# 或直接复制到 Windows 能访问的 SD 卡盘符下
```

### 5.3 验证传输成功

```bash
ssh root@192.168.42.1
ls -la /root/huling_*
# 应看到三个文件，大小 > 100KB
```

---

## 阶段 6：PC 端准备测试数据

> 回到 Windows PowerShell / cmd 中执行。

### 6.1 确认 Python 环境

```cmd
python --version
# Python 3.8+

pip list | findstr mediapipe
# 如果没有: pip install mediapipe opencv-python

pip list | findstr numpy
# 如果没有: pip install numpy
```

### 6.2 确认摄像头工作

```cmd
python -c "import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
# 输出: OK
```

### 6.3 运行关键点采集

```cmd
cd D:\Users\wangxianxiu\.openclaw\workspace\huling_model\deploy\duos

python keypoint_bridge.py --input camera --output file --path test_kps.txt --max-frames 150
```

程序会打开摄像头窗口，你需要：
```
① 前 30 帧：正常站立（让模型看到 walking）
② 第 30-60 帧：坐下（让模型看到 sitting）
③ 第 60-90 帧：站起来走动
④ 第 90-120 帧：模拟跌倒（从站立快速蹲下/倒下）
⑤ 第 120-150 帧：恢复正常站立
```

按 `q` 可以提前结束。

### 6.4 检查生成的文件

```cmd
type test_kps.txt | more
```

应看到 33×150 = 4950 行关键点数据 + 150 行分隔符 `---`。

每行格式：
```
{"x":0.500,"y":0.180,"z":0.000,"v":0.950}
```

### 6.5 传输到 Duo S

```cmd
# 在 PowerShell 中用 scp（需要先安装 OpenSSH 客户端）
scp test_kps.txt root@192.168.42.1:/root/
# 密码: milkv
```

或者使用 WinSCP / FileZilla（图形界面拖拽）。

---

## 阶段 7：在 Duo S 上推理验证

### 7.1 登录 Duo S

```cmd
ssh root@192.168.42.1
```

### 7.2 运行自检程序

```bash
cd /root
./huling_test
```

**预期输出：**
```
========================================
  HuLing C Deployment Verification
  Target: Milk-V Duo S
========================================
  Test cases: 6
  Features: 98

--- Case 0: standing ---
  Features: OK (98/98 match)
  C predict: 0 (walking)  Py predict: 0
  Normalized: OK (98/98 match)

--- Case 1: sitting ---
  Features: OK (98/98 match)
  C predict: 1 (sitting)  Py predict: 1
  Normalized: OK (98/98 match)

--- Case 2: lying ---
  Features: OK (98/98 match)
  C predict: 5 (fall)  Py predict: 5
  Normalized: OK (98/98 match)
  ...

========================================
  ALL TESTS PASSED
========================================
```

> ℹ️ 注意：因为模型只训练了 3 类（0=walking, 1=sitting, 5=fall），躺姿 (lying) 会被归类为 fall。

### 7.3 在测试数据上推理

```bash
./huling_demo < test_kps.txt
```

**预期输出：**
```
========================================
  HuLing Pose Classifier for Duo S
  Model: RandomForest 100-tree, 98-dim
  Classes: walking/sitting/lying/long_sit/abnormal/fall
========================================

[Frame    0]    walking  conf=0.892  (0.89 0.05 0.03 0.00 0.00 0.03)
[Frame    1]    walking  conf=0.867  (0.87 0.07 0.03 0.00 0.00 0.03)
[Frame    2]    walking  conf=0.901  (0.90 0.04 0.03 0.00 0.00 0.03)
...
[Frame   35]    sitting  conf=0.876  (0.06 0.88 0.03 0.00 0.00 0.03)
[Frame   36]    sitting  conf=0.914  (0.02 0.91 0.04 0.00 0.01 0.03)
...
[Frame   95]       fall  conf=0.789  (0.05 0.06 0.05 0.00 0.05 0.79)
[Frame   96]       fall  conf=0.823  (0.03 0.04 0.06 0.00 0.04 0.82)
...
========================================
  Summary
========================================
  Total frames:   150
  Total time:     0.235 s
  Avg inference:  1.57 ms/frame
  FPS:            638.3

  Class distribution:
    walking   :   60 ( 40.0%)
    sitting   :   45 ( 30.0%)
    lying     :    0 (  0.0%)
    long_sit  :    0 (  0.0%)
    abnormal  :    0 (  0.0%)
    fall      :   35 ( 23.3%)
```

### 7.4 分析结果

对照你在摄像头前做的动作，检查分类是否正确：
- 站立/走动 → 应为 **walking**
- 坐着 → 应为 **sitting**
- 跌倒 → 应为 **fall**
- 如果靠近跌倒且置信度不高，说明模型需要更多训练数据

---

## 阶段 8：实时网络推理

### 8.1 在 Duo S 上启动 UDP 推理服务器

```bash
# SSH 到 Duo S
ssh root@192.168.42.1

cd /root
./huling_server
```

输出：
```
========================================
  HuLing Real-time Server for Duo S
  Listening on UDP port 8888
========================================
[INFO] Ready. Waiting for keypoint packets...
[INFO] Start PC: python keypoint_bridge.py --output udp --host 192.168.42.1 --port 8888
```

### 8.2 在 PC 上启动关键点发送

另开一个 cmd/PowerShell 窗口：

```cmd
cd D:\Users\wangxianxiu\.openclaw\workspace\huling_model\deploy\duos

python keypoint_bridge.py --input camera --output udp --host 192.168.42.1 --port 8888
```

### 8.3 实时效果

**PC 窗口**：显示摄像头画面 + 黄色骨架关键点 + 帧计数

**Duo S SSH 窗口**：实时打印每帧分类结果
```
[    0]    walking conf=0.901 (  1.2ms)
[    1]    walking conf=0.888 (  1.1ms)
[    2]    walking conf=0.902 (  0.9ms)
[   45]    sitting conf=0.867 (  1.0ms)
[   46]    sitting conf=0.891 (  1.1ms)
[   90]       fall conf=0.756 (  1.2ms)
  *** FALL DETECTED! ***
[   91]       fall conf=0.812 (  1.0ms)
  *** FALL DETECTED! ***
```

按 `Ctrl+C` 停止 PC 端和 Duo S 端。

### 8.4 性能指标

正常情况下：
- **PC 端 MediaPipe**：20-30ms/帧 (33-50 FPS)
- **UDP 传输延迟**：<1ms（本地网络）
- **Duo S 推理**：<2ms/帧 (500+ FPS)
- **端到端延迟**：约 25-35ms

---

## 阶段 9：进阶 — Duo S 独立运行（去掉 PC）

当前方案中，MediaPipe 在 PC 上运行。要让 Duo S 完全独立运行，需要把姿态估计也迁移到 Duo S 上。

### 9.1 方案选择

| 方案 | 帧率 | 功耗 | 实现难度 |
|------|------|------|---------|
| CVITEK TPU person_keypoint | 15-25 FPS | 低 | ⭐⭐⭐ |
| OpenCV DNN + MoveNet (CPU) | 5-10 FPS | 中 | ⭐⭐ |
| 继续用 PC + Duo S | 30 FPS | — | ⭐(已完成) |

### 9.2 CVITEK TPU 方案（推荐）

Duo S 的 SG2000 芯片内置 TPU，SDK 的 `cvi_tdl` 框架提供人体关键点检测。

```bash
# 在 Duo S 的 Buildroot SDK 中
cd $MILKV_SDK

# 进入 buildroot 配置
cd buildroot-2024.02
make milkv_duos_defconfig

# 启用 TDL SDK
make menuconfig
# 导航到: External options → Cvitek TDL SDK → [*] Enable
# 保存退出

# 重新编译系统
make
```

更多 TDL SDK 使用方法参考：  
https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x

### 9.3 重新训练完整 6 类模型（可选）

当前模型只训练了 3 类（walking/sitting/fall）。如果需要完整的 6 类检测：

```cmd
cd D:\Users\wangxianxiu\.openclaw\workspace\huling_model

# 用 data_capture.py 录制更多数据（lying / long_sit / abnormal）
python collect_data.py

# 重新训练
python train_model.py --csv data/huling_data_xxx.csv

# 重新导出 C 代码
python export_to_c.py
```

---

## 附录A：文件结构速查

```
huling_model/
├── models/
│   └── pose_classifier.joblib          ← 训练好的 Python 模型 (3类: walking/sitting/fall)
│
├── deploy/                              ← 部署相关
│   ├── huling_deploy.h                 ← C API 头文件
│   ├── huling_features.c               ← 98维特征提取 (430行)
│   ├── scaler_params.h                 ← StandardScaler mean/std (98参数)
│   ├── random_forest.c                 ← RandomForest 100棵树 (84000行, m2cgen生成)
│   ├── random_forest.h                 ← RandomForest 头文件
│   ├── random_forest_wrapper.c         ← float→double桥接 + argmax
│   ├── test_data.h                     ← 测试参考数据
│   ├── test_deploy.c                   ← PC验证程序
│   │
│   └── duos/                           ← 🎯 Duo S 专用部署包
│       ├── DEPLOY.md                   ← 本文档
│       ├── CMakeLists.txt              ← CMake构建文件
│       ├── toolchain-riscv.cmake        ← RISC-V 交叉编译配置
│       ├── toolchain-arm.cmake          ← ARM 交叉编译配置
│       ├── main.c                      ← 演示程序 (stdin→分类)
│       ├── udp_server.c                ← UDP实时推理服务器
│       ├── test_deploy_duos.c          ← 板载验证程序
│       └── keypoint_bridge.py          ← PC端关键点采集+发送
```

## 附录B：常用命令速查

```bash
# === Duo S 操作 ===
ssh root@192.168.42.1                    # 登录
scp 文件 root@192.168.42.1:/root/       # 传文件

# === 编译相关 ===
export MILKV_SDK=$HOME/duo-buildroot-sdk
cd deploy/duos/build-riscv
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv.cmake && make

# === 推理 ===
./huling_demo < test_kps.txt             # 文件模式
./huling_server                          # UDP实时模式
./huling_test                            # 自检验证

# === PC端关键点采集 ===
python keypoint_bridge.py --input camera --output file --path data.txt
python keypoint_bridge.py --input camera --output udp --host 192.168.42.1
```

## 附录C：完整依赖清单

| 工具 | 版本要求 | 用途 |
|------|---------|------|
| Python | 3.8+ | 关键点提取/训练 |
| MinGW-w64 | 任意 | PC端 C 代码编译 |
| WSL2 / Linux | Ubuntu 20.04+ | SDK 编译 |
| cmake | 3.10+ | 交叉编译配置 |
| gcc (RISC-V) | 随 SDK | Duo S 交叉编译 |
| mediapipe | 0.10+ | 姿态关键点提取 |
| scikit-learn | 1.3+ | 模型训练 |
| m2cgen | 0.10+ | 模型→C代码转换 |
