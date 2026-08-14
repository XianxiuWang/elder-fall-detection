# SenseVoice 模型下载教程

> 适用环境：Windows x64 / Linux x64  
> 用途：下载 SenseVoice 预训练模型（PyTorch 版）+ ONNX 导出版 + 源码仓库

---

## 一、SenseVoice 简介

SenseVoice 是阿里 FunAudioLLM 团队开源的多语言语音理解模型，支持：

| 能力 | 说明 |
|------|------|
| 语音识别 (ASR) | 中/英/日/韩/粤，中文识别精度极高 |
| 语言检测 | 自动识别语种 |
| 情绪识别 | 识别说话人情绪（中性/高兴/悲伤/愤怒等） |
| 音频事件检测 | 检测掌声、笑声、背景音乐等 |
| 逆文本正则化 (ITN) | "九点" → "9点" 自动转换 |

有两个版本：

| 版本 | 大小 | 说明 |
|------|------|------|
| SenseVoiceSmall | ~900MB (float32) / ~230MB (int8) | **推荐**，精度高，速度快 |
| SenseVoiceLarge | ~3GB | 精度最高，但资源消耗大 |

---

## 二、方式 1：下载预训练 ONNX 模型（推荐，直接能用）

> 适合：想直接用 sherpa-onnx 做推理，不需要训练

### 2.1 int8 量化版（228MB，推荐）

```bash
# Linux / WSL2 / Git Bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
```

解压后文件：

```
sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/
├── model.int8.onnx       # 228MB，量化模型
├── tokens.txt            # 词表文件
├── test_wavs/            # 测试音频
│   ├── zh.wav
│   ├── en.wav
│   ├── ja.wav
│   ├── ko.wav
│   └── yue.wav
├── export-onnx.py        # ONNX 导出脚本
├── LICENSE
└── README.md
```

### 2.2 float32 完整版（894MB）

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
```

解压后多一个 `model.onnx`（894MB）。

### 2.3 Windows 下没有 wget 怎么办？

**方法 A：浏览器直接下载**

打开以下链接下载 `.tar.bz2` 文件，然后用 7-Zip 解压两次（先解 `.bz2`，再解 `.tar`）：

- int8 版：`https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2`

- float32 版：`https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2`

**方法 B：安装 wget for Windows**

```powershell
# PowerShell 管理员权限
winget install GNU.Wget2
# 或下载: https://eternallybored.org/misc/wget/
```

**方法 C：用 PowerShell 下载**

```powershell
# 下载 int8 版
Invoke-WebRequest -Uri "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2" -OutFile "sensevoice_int8.tar.bz2"
```

---

## 三、方式 2：用 Python / modelscope 下载 PyTorch 模型

> 适合：需要训练/微调模型

### 3.1 安装依赖

```bash
pip install modelscope funasr
```

### 3.2 下载 SenseVoiceSmall

```python
# download_sensevoice.py
from modelscope import snapshot_download

# 下载 SenseVoiceSmall
model_dir = snapshot_download(
    'iic/SenseVoiceSmall',
    cache_dir='./models'  # 保存到当前目录的 models/ 下
)
print(f"✅ 模型已下载到: {model_dir}")
```

运行：

```bash
python download_sensevoice.py
```

下载后目录结构：

```
models/iic/SenseVoiceSmall/
├── model.pt              # PyTorch 模型权重
├── config.yaml           # 模型配置
├── tokens.txt            # 词表
├── am.mvn                # 均值方差归一化参数
├── configuration.json    # 训练配置
└── README.md
```

### 3.3 下载 SenseVoiceLarge（可选）

```python
from modelscope import snapshot_download

model_dir = snapshot_download(
    'iic/SenseVoiceLarge',
    cache_dir='./models'
)
print(f"✅ 模型已下载到: {model_dir}")
```

---

## 四、方式 3：克隆源码仓库

### 4.1 SenseVoice 官方仓库

```bash
git clone https://github.com/FunAudioLLM/SenseVoice.git
cd SenseVoice
pip install -r requirements.txt
```

> 国内加速：`git clone https://gitee.com/mirrors/SenseVoice.git`

### 4.2 sherpa-onnx 仓库（含导出脚本）

```bash
git clone https://github.com/k2-fsa/sherpa-onnx.git
# 导出脚本位置: sherpa-onnx/scripts/sense-voice/export-onnx.py
```

---

## 五、下载后验证

### 5.1 用 sherpa-onnx 验证 ONNX 模型

```bash
# 安装 sherpa-onnx
pip install sherpa-onnx

# 测试中文识别
sherpa-onnx-offline \
  --tokens=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/tokens.txt \
  --sense-voice-model=./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/model.int8.onnx \
  --num-threads=4 \
  --sense-voice-use-itn=1 \
  ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/test_wavs/zh.wav
```

预期输出：

```json
{
  "lang": "<|zh|>",
  "emotion": "<|NEUTRAL|>",
  "event": "<|Speech|>",
  "text": "开放时间早上9点至下午5点。",
  "timestamps": [...],
  "tokens": ["开","放","时","间","早","上","9","点","至","下","午","5","点","。"]
}
```

### 5.2 用 Python 验证 PyTorch 模型

```python
# test_sensevoice.py
from funasr import AutoModel

model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
)

result = model.generate(
    input="test.wav",
    language="zh",
    use_itn=True,
)
print(result)
```

---

## 六、各版本对比总结

| 下载方式 | 格式 | 大小 | 用途 |
|----------|------|------|------|
| sherpa-onnx int8 | model.int8.onnx | 228MB | **直接推理**（推荐） |
| sherpa-onnx float32 | model.onnx | 894MB | 高精度推理 |
| modelscope | model.pt | ~900MB | **微调训练** |
| GitHub 源码 | Python 代码 | ~50MB | 自定义开发 |

---

## 七、快速下载脚本（一键）

保存以下内容为 `download_sensevoice.sh`：

```bash
#!/bin/bash
echo "=== SenseVoice 模型一键下载 ==="

# ONNX int8 版（推荐）
echo "[1/2] 下载 ONNX int8 模型..."
wget -q --show-progress \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
tar xf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
echo "✅ ONNX int8 模型下载完成"

# PyTorch 版（用于训练）
echo "[2/2] 下载 PyTorch 模型..."
pip install modelscope -q
python3 -c "
from modelscope import snapshot_download
d = snapshot_download('iic/SenseVoiceSmall', cache_dir='./models')
print(f'✅ PyTorch 模型已下载到: {d}')
"

echo ""
echo "=== 下载完成 ==="
echo "ONNX 模型: ./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17/"
echo "PyTorch 模型: ./models/iic/SenseVoiceSmall/"
```

Windows PowerShell 版（`download_sensevoice.ps1`）：

```powershell
Write-Host "=== SenseVoice 模型一键下载 ===" -ForegroundColor Green

# 下载 ONNX int8 版
Write-Host "[1/2] 下载 ONNX int8 模型..." -ForegroundColor Yellow
$url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
$out = "sensevoice_int8.tar.bz2"
Invoke-WebRequest -Uri $url -OutFile $out
Write-Host "下载完成，请用 7-Zip 解压 $out" -ForegroundColor Green

# 下载 PyTorch 模型
Write-Host "[2/2] 下载 PyTorch 模型..." -ForegroundColor Yellow
pip install modelscope -q
python -c "from modelscope import snapshot_download; d=snapshot_download('iic/SenseVoiceSmall',cache_dir='./models'); print(f'PyTorch模型: {d}')"
Write-Host "✅ 全部完成" -ForegroundColor Green
```

---

> 📅 最后更新：2026-07-16  
> 📄 版本：sherpa-onnx v1.13.x / SenseVoice 2024-07-17
