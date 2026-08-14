# HuLing 部署到 Milk-V Duo S — 完整指南

## 概述

本项目是一个基于姿态关键点的人体状态分类器（护龄），使用 RandomForest 对 98 维特征进行 6 类分类。

本指南描述如何将训练好的模型部署到 Milk-V Duo S (SG2000) 开发板。

---

## 模型架构（已完成的 C 代码）

```
┌──────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────┐
│ 33关键点  │───▶│ 特征提取 (98维)  │───▶│ StandardScaler │───▶│RandomForest│───▶ class
│ (x,y,z,v)│    │ huling_features.c│    │scaler_params.h│    │random_forest.c│   (0-5)
└──────────┘    └─────────────────┘    └──────────────┘    └──────────┘
```

| 文件 | 说明 |
|------|------|
| `deploy/huling_deploy.h` | 统一 API 头文件 |
| `deploy/huling_features.c` | 特征提取（6大模块，98维），精确复现 Python |
| `deploy/scaler_params.h` | StandardScaler 的 mean/std 参数 |
| `deploy/random_forest.c` | 100 棵树的 RandomForest（m2cgen 生成，约 84000 行） |
| `deploy/random_forest.h` | RandomForest 头文件 |
| `deploy/random_forest_wrapper.c` | float→double 桥接 + argmax 包装 |
| `deploy/test_data.h` | 测试数据（5种姿态的参考值） |

**当前模型训练了 3 个类别：0=walking, 1=sitting, 5=fall**

---

## 部署步骤

### 第1步：PC 端验证（确认 C 代码与 Python 一致）

```bash
# 在 PC 上编译测试（MinGW/MSVC）
cd D:\Users\wangxianxiu\.openclaw\workspace\huling_model\deploy

# MinGW:
gcc -o test_deploy.exe huling_features.c random_forest.c random_forest_wrapper.c test_deploy.c -lm -O0 -Wall

# 运行验证
./test_deploy.exe
```

预期输出：**ALL TESTS PASSED**（特征提取+标准化+推理均与Python一致）

### 第2步：获取 Milk-V Duo S SDK

```bash
# 克隆 Buildroot SDK
git clone https://github.com/milkv-duo/duo-buildroot-sdk.git
cd duo-buildroot-sdk

# 构建（生成交叉编译工具链）
./build.sh milkv-duos-sd
```

编译完成后，工具链路径：
- RISC-V: `duo-buildroot-sdk/host-tools/gcc/riscv64-linux-musl-x86_64/bin/`
- ARM: `duo-buildroot-sdk/host-tools/gcc/arm-buildroot-linux-gnueabihf/bin/`

### 第3步：交叉编译 C 代码

```bash
# 设置 SDK 路径
export MILKV_SDK=/path/to/duo-buildroot-sdk

# 编译 RISC-V 版本
cd duos/
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv.cmake
make

# 产物：
#   libhuling.a        — 静态库
#   huling_demo        — 演示程序（从stdin读取关键点）
#   huling_test        — 验证程序
```

### 第4步：部署到 Duo S

```bash
# 方式1: 通过 SSH 复制（WiFi/以太网）
scp huling_demo root@192.168.x.x:/root/
scp test_deploy root@192.168.x.x:/root/

# 方式2: 复制到 SD 卡
cp huling_demo /mnt/sdcard/

# SSH 登录 Duo S
ssh root@192.168.x.x
cd /root
./huling_test   # 先验证结果
```

### 第5步：生成关键点数据并在 Duo S 上推理

**PC 端（提取关键点）：**
```bash
# 安装依赖
pip install mediapipe opencv-python

# 从摄像头提取关键点，保存为文本文件
cd deploy/duos/
python keypoint_bridge.py --input camera --output file --path test_kps.txt --max-frames 100

# 复制到 Duo S
scp test_kps.txt root@192.168.x.x:/root/
```

**Duo S 端（推理）：**
```bash
./huling_demo < test_kps.txt
```

### 第6步（进阶）：实时网络推理

**PC 端：**
```bash
python keypoint_bridge.py --input camera --output udp --host 192.168.x.x --port 8888
```

**Duo S 端需要 UDP 接收程序**（见 `udp_server.c` 示例）

---

## 关键限制：姿态估计

部署的最大挑战是 **Duo S 上缺少 MediaPipe**。C 代码只实现分类推理，姿态关键点需要外部输入。

### 三种方案对比

| 方案 | 姿态估计位置 | 延迟 | 复杂度 | 推荐 |
|------|------------|------|--------|------|
| **A: PC提取+网络发送** | PC (MediaPipe) | ~50ms | 低 | 快速验证 ✅ |
| **B: Duo S TPU+YOLO-Pose** | Duo S TPU | ~100ms | 中 | 独立运行 ⭐ |
| **C: Duo S CPU+MoveNet** | Duo S CPU | ~200ms | 中 | 无TPU替代 |

### 方案A（推荐入门）：已经完成

PC 运行 `keypoint_bridge.py` → UDP/文件 → Duo S 运行 `huling_demo`

### 方案B（进阶）：使用 CVITEK TPU

Milk-V Duo S 的 SDK 提供 `cvi_tdl` (CVITEK TDL SDK)，支持：
- `cvi_tdl_person_detection` — 人体检测
- `cvi_tdl_person_keypoint` — 人体关键点检测
- 模型运行在 0.5TOPS TPU 上，功耗低

需要：
1. 使用 `cvitek_tdl_sdk` 获取 17 个人体关键点
2. 将 17 个关键点映射为 33 个 MediaPipe 格式（见 `keypoint_bridge.py` 中的 MoveNet→MediaPipe 映射）
3. 调用 `huling_predict()` 进行分类

### 方案C：OpenCV DNN + MoveNet on Duo S CPU

在 Duo S Buildroot 中启用 OpenCV + DNN 模块，运行 MoveNet Lightning TFLite 模型。

---

## 模型优化建议

### 1. 减少特征维度（减小模型）
当前模型使用 98 维特征，但 RandomForest 只用到了约 30 个有效特征。可以：
- 分析 `feature_extractor.feature_importances_` 取 Top-30
- 重新训练只用这些特征的模型
- C 代码体积可从 84000 行降至 ~25000 行

### 2. 减少树的数量
从 100 棵树减到 50 棵，精度损失 <2%，代码体积减半：
```python
# config.py
RF_N_ESTIMATORS = 50
RF_MAX_DEPTH = 10
```

### 3. 训练 6 类完整模型
当前模型只有 3 类（walking/sitting/fall），建议补充：
- lying（躺卧）
- long_sit（久坐）
- abnormal（异常姿态）

---

## 文件清单

```
deploy/
├── huling_deploy.h          # 统一 API 头
├── huling_features.c         # 特征提取 C 实现
├── scaler_params.h           # StandardScaler 参数
├── random_forest.c           # RandomForest (m2cgen, 100 trees)
├── random_forest.h           # RandomForest 头文件
├── random_forest_wrapper.c   # 包装函数
├── test_data.h               # 测试数据
├── test_deploy.c             # PC 验证程序
│
└── duos/                     # ← Duo S 专用部署包
    ├── CMakeLists.txt        # CMake 构建
    ├── toolchain-riscv.cmake # RISC-V 交叉编译配置
    ├── toolchain-arm.cmake   # ARM 交叉编译配置
    ├── main.c                # 演示程序 (stdin/JSON → 分类)
    ├── test_deploy_duos.c    # 板载验证程序
    └── keypoint_bridge.py    # PC端关键点提取 → 发送给Duo S
```

---

## 参考链接

- Milk-V Duo S 官方文档: https://milkv.io/docs/duo/overview
- Duo Buildroot SDK: https://github.com/milkv-duo/duo-buildroot-sdk
- CVITEK TDL SDK: https://github.com/milkv-duo/cvitek-tdl-sdk-sg200x
- m2cgen (模型→C代码): https://github.com/BayesWitnesses/m2cgen
