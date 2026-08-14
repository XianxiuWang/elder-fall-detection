# SenseVoice 模型离线安装与分发教程

> 场景：A 电脑下载模型 → 压缩打包 → 发给 B 电脑直接使用  
> 优势：B 不需要联网下载 900MB 模型，解压即用

---

## 第一步：A 电脑 — 下载模型

在已配置好网络的 A 电脑上：

```cmd
:: 1. 创建 conda 环境
conda create -n sensevoice python=3.10 -y
conda activate sensevoice

:: 2. 安装 modelscope（下载工具）
pip install modelscope

:: 3. 下载 SenseVoiceSmall 模型到指定目录
python -c "from modelscope import snapshot_download; d=snapshot_download('iic/SenseVoiceSmall', cache_dir='E:/SenseVoice_offline'); print(f'模型已下载到: {d}')"
```

等几分钟，看到 `模型已下载到: E:/SenseVoice_offline/iic/SenseVoiceSmall` 就完成了。

---

## 第二步：A 电脑 — 打包压缩

模型目录太大了（~900MB），需要压缩：

### 方式 1：用 tar 打包（推荐，跨平台）

```cmd
:: Windows 11 自带 tar，Win10 需要先装
tar -czf E:/sensevoice_model.tar.gz -C E:/SenseVoice_offline iic
```

打包后 `E:/sensevoice_model.tar.gz` 约 850MB，比原文件小一点。

### 方式 2：用 7-Zip 打包（Windows 通用）

右键 `E:\SenseVoice_offline` 文件夹 → 7-Zip → 添加到压缩包 → 格式选 `zip` → 压缩级别选"极限"。

打包后 `E:/sensevoice_model.zip` 约 870MB。

### 方式 3：用 PowerShell 打包

```powershell
Compress-Archive -Path "E:\SenseVoice_offline\*" -DestinationPath "E:\sensevoice_model.zip"
```

---

## 第三步：传输给 B 电脑

打包文件约 850MB，任选一种方式发给 B：

| 方式 | 操作 |
|------|------|
| **百度网盘** | 上传 `sensevoice_model.zip`，分享链接给 B |
| **阿里云盘** | 上传压缩包，不限速，国内推荐 |
| **QQ/微信** | 发文件，但 850MB 可能超限 |
| **U 盘对拷** | 最原始但最可靠 |
| **局域网直传** | A 和 B 都在同一个 WiFi 下：`python -m http.server 8000` 然后 B 浏览器访问 `http://A的IP:8000` 下载 |

---

## 第四步：B 电脑 — 接收并安装

### 4.1 接收压缩包

B 收到 `sensevoice_model.zip` 后，放到 `E:\` 目录下。

### 4.2 解压

```cmd
:: 如果收到的是 zip
cd /d E:\
tar -xf sensevoice_model.zip

:: 或用 PowerShell
powershell Expand-Archive -Path "E:\sensevoice_model.zip" -DestinationPath "E:\"
```

解压后目录结构：

```
E:\SenseVoice_offline\
└── iic\
    └── SenseVoiceSmall\
        ├── model.pt              # PyTorch 模型权重
        ├── config.yaml           # 模型配置
        ├── configuration.json    # 训练配置
        ├── tokens.txt            # 词表
        ├── am.mvn                # 归一化参数
        └── README.md
```

### 4.3 创建环境并安装依赖

```cmd
:: 创建 conda 环境
conda create -n sensevoice python=3.10 -y
conda activate sensevoice

:: 安装 PyTorch（根据 B 的显卡选版本）
:: 有 NVIDIA 显卡：
conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
:: 没显卡：
:: conda install pytorch torchaudio cpuonly -c pytorch -y

:: 安装 FunASR 训练框架
pip install funasr
```

### 4.4 验证模型

```cmd
python -c "
from funasr import AutoModel

# 用本地路径加载模型（关键！注意路径格式）
model = AutoModel(
    model='E:/SenseVoice_offline/iic/SenseVoiceSmall',
    trust_remote_code=True,
    device='cuda:0',  # 有显卡用这个；没显卡改为 'cpu'
)
print('✅ 模型加载成功')
print(model)
"
```

> ⚠️ 路径格式：Windows 下用 `E:/SenseVoice_offline/iic/SenseVoiceSmall`（斜杠正写）

---

## 第五步：B 电脑 — 一键安装脚本

把下面内容保存为 `E:\setup_sensevoice.bat`，发给 B，双击运行：

```batch
@echo off
echo ========================================
echo   SenseVoice 模型离线安装
echo ========================================

:: 创建环境
echo [1/3] 创建 conda 环境...
call conda create -n sensevoice python=3.10 -y

:: 激活环境
call conda activate sensevoice

:: 安装 PyTorch（有显卡版）
echo [2/3] 安装 PyTorch...
call conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

:: 安装 FunASR
echo [3/3] 安装 FunASR...
pip install funasr

:: 验证
echo.
echo ========================================
echo   验证模型...
echo ========================================
python -c "from funasr import AutoModel; m=AutoModel(model='E:/SenseVoice_offline/iic/SenseVoiceSmall', trust_remote_code=True); print('✅ 安装成功！模型就绪')"

echo.
echo 全部完成！
pause
```

---

## 快速检查清单

| 步骤 | A 电脑 | B 电脑 |
|------|--------|--------|
| ✅ 下载 modelscope | `pip install modelscope` | 不需要 |
| ✅ 下载模型 | `snapshot_download()` | 不需要，收压缩包就行 |
| ✅ 打包 | `tar/zip` 压缩 | 不需要 |
| ✅ 传输 | 百度网盘 / U盘 | 接收文件 |
| ✅ 安装环境 | 不需要 | `conda create + pip install` |
| ✅ 验证 | 不需要 | `AutoModel(model='本地路径')` 加载 |

---

## 常见问题

| 问题 | 解决 |
|------|------|
| 解压后文件不全 | 检查压缩包是否完整下载（~850MB），重传 |
| `AutoModel` 加载报找不到模型 | 路径写成 `E:/SenseVoice_offline/iic/SenseVoiceSmall`，**不要用反斜杠** |
| B 电脑没 conda | 先去 https://docs.conda.io/en/latest/miniconda.html 下载 Miniconda |
| B 电脑没显卡 | PyTorch 装 CPU 版：`conda install pytorch torchaudio cpuonly -c pytorch -y` |
| 压缩包太大传不动 | 用 `tar -czf` 命令（比 zip 小 20MB），或者分卷压缩 |

---

> 📅 最后更新：2026-07-16  
> 📦 模型：SenseVoiceSmall (iic/SenseVoiceSmall)  
> 🔧 框架：FunASR + PyTorch
