# SenseVoice 微调训练教程

> 环境：Windows + WSL2 Ubuntu 22.04（推荐）/ 纯 Linux x64  
> GPU：NVIDIA GTX 1060 6GB 及以上  
> 用途：用自己的数据微调 SenseVoiceSmall，适配特定领域/口音/场景

---

## 一、环境搭建

### 1.1 安装 WSL2 + CUDA（Windows 用户）

```powershell
# PowerShell（管理员身份）
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```

重启后进入 Ubuntu，完成初始设置，然后安装 CUDA：

```bash
# WSL2 内安装 CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-3

# 验证 CUDA
nvidia-smi
```

### 1.2 安装 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 按提示操作，安装完重启终端
```

### 1.3 创建训练环境

```bash
# 创建 conda 环境
conda create -n sensevoice python=3.10 -y
conda activate sensevoice

# 安装 PyTorch（CUDA 12.1）
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 验证 PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 克隆 SenseVoice 仓库
git clone https://github.com/FunAudioLLM/SenseVoice.git
cd SenseVoice

# 安装 FunASR 和相关依赖
pip install funasr modelscope
pip install -r requirements.txt

# 额外工具
pip install onnx onnxruntime soundfile librosa
```

---

## 二、准备训练数据

### 2.1 目录结构

```
SenseVoice/
└── data/
    ├── audio_16k/          # 16kHz WAV 音频
    │   ├── sample_001.wav
    │   ├── sample_002.wav
    │   └── ...
    ├── train.jsonl         # 训练集
    └── val.jsonl           # 验证集（可选，建议从训练集抽 10%）
```

### 2.2 音频格式要求

| 参数 | 要求 |
|------|------|
| 格式 | WAV |
| 采样率 | 16000 Hz |
| 位深度 | 16 bit |
| 声道 | 单声道 (mono) |
| 时长 | 建议 2-20 秒/段 |

### 2.3 批量音频转换

```bash
# 安装 ffmpeg
sudo apt install -y ffmpeg

# 批量转换：mp3/m4a/flac → 16kHz mono WAV
mkdir -p data/raw_audio data/audio_16k

for f in data/raw_audio/*; do
    name=$(basename "$f" | sed 's/\.[^.]*$//')
    ffmpeg -i "$f" -ar 16000 -ac 1 -sample_fmt s16 "data/audio_16k/${name}.wav" -y -loglevel error
    echo "✅ ${name}.wav"
done

echo "转换完成！共 $(ls data/audio_16k/*.wav | wc -l) 个文件"
```

### 2.4 数据标注格式

**`data/train.jsonl`**，每行一个 JSON：

```jsonl
{"key": "sample_001", "source": "data/audio_16k/sample_001.wav", "target": "今天天气真好我们出去走走吧", "source_lang": "zh"}
{"key": "sample_002", "source": "data/audio_16k/sample_002.wav", "target": "请帮我查一下明天的航班信息", "source_lang": "zh"}
{"key": "sample_003", "source": "data/audio_16k/sample_003.wav", "target": "这个功能需要在配置文件中设置参数", "source_lang": "zh"}
{"key": "sample_004", "source": "data/audio_16k/sample_004.wav", "target": "病人血压一百四九十需要立即处理", "source_lang": "zh"}
{"key": "sample_005", "source": "data/audio_16k/sample_005.wav", "target": "下午三点的会议室已经预约好了", "source_lang": "zh"}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `key` | string | 唯一标识 |
| `source` | string | 音频文件路径（相对路径） |
| `target` | string | 标注文本（不要标点，ITN 会自动加） |
| `source_lang` | string | 语言：zh / en / ja / ko / yue |

### 2.5 生成数据清单脚本

```python
# generate_manifest.py
"""
从 transcriptions.txt + audio_16k/ 生成 train.jsonl 和 val.jsonl

transcriptions.txt 格式（每行）：
    文件名（不含扩展名）<TAB>标注文本
    示例：sample_001	今天天气真好我们出去走走吧
"""
import json
import os
import random

audio_dir = "data/audio_16k"
manifest_file = "data/transcriptions.txt"
train_output = "data/train.jsonl"
val_output = "data/val.jsonl"
val_ratio = 0.1  # 10% 作为验证集

data_list = []
with open(manifest_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            filename, text = parts[0], parts[1]
            data_list.append({
                "key": filename,
                "source": os.path.join(audio_dir, f"{filename}.wav"),
                "target": text,
                "source_lang": "zh"
            })

# 随机打乱
random.shuffle(data_list)

# 分割训练/验证
split_idx = int(len(data_list) * (1 - val_ratio))
train_data = data_list[:split_idx]
val_data = data_list[split_idx:]

# 写入
for path, data in [(train_output, train_data), (val_output, val_data)]:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 训练集: {len(train_data)} 条 -> {train_output}")
print(f"✅ 验证集: {len(val_data)} 条  -> {val_output}")
```

运行：

```bash
python generate_manifest.py
```

### 2.6 数据量建议

| 场景 | 最少 | 推荐 | 备注 |
|------|------|------|------|
| 轻量领域适配 | 500 条 (~1h) | 2000 条 | 只微调顶层 |
| 中度定制 | 2000 条 (~4h) | 5000 条 | 微调后半段 |
| 深度定制 | 5000 条 (~10h) | 10000+ 条 | 开放更多层训练 |
| 全新口音/方言 | 3000 条 | 8000+ 条 | 可能需要更多 epoch |

---

## 三、配置训练参数

### 3.1 创建配置文件

```bash
mkdir -p conf
```

创建 `conf/finetune.yaml`：

```yaml
# ============================================
# SenseVoice Small 微调配置
# 适用场景：中文 ASR 领域适配
# ============================================

# --- 模型 ---
model: SenseVoiceSmall
model_dir: iic/SenseVoiceSmall      # 自动从 modelscope 下载
# 如果已下载，改为本地路径：
# model_dir: /path/to/models/iic/SenseVoiceSmall

# --- 数据 ---
train_data: data/train.jsonl
valid_data: data/val.jsonl

# --- 训练超参数 ---
batch_size: 4              # GPU 显存不够就减小
accum_grad: 4              # 梯度累积（等效 batch_size = 4 × 4 = 16）
max_epoch: 20
learning_rate: 0.0001      # 1e-4
warmup_steps: 500
grad_clip: 5.0

# --- 优化器 ---
optim: adamw
optim_conf:
  lr: 0.0001
  betas: [0.9, 0.999]
  weight_decay: 0.0001

# --- 调度器 ---
scheduler: warmuplr
scheduler_conf:
  warmup_steps: 500

# --- 模型冻结策略 ---
# 如果数据 < 1000 条，建议冻结更多层
freeze_encoder: false       # 数据少时设为 true
freeze_frontend: true       # 前端特征提取保持冻结

# --- 输出 ---
output_dir: ./output/sensevoice_finetuned
save_checkpoint_interval: 1000
log_interval: 50

# --- 其他 ---
seed: 42
use_amp: true               # 混合精度训练
num_workers: 4
```

### 3.2 根据 GPU 显存调整 batch_size

| GPU 显存 | batch_size | accum_grad | 等效 batch |
|----------|-----------|------------|------------|
| 4GB | 2 | 8 | 16 |
| 6GB | 4 | 4 | 16 |
| 8GB | 8 | 2 | 16 |
| 12GB+ | 16 | 1 | 16 |

---

## 四、启动训练

### 4.1 快速启动命令

```bash
cd ~/SenseVoice
conda activate sensevoice

python funasr/bin/train.py \
  --config-path conf \
  --config-name finetune.yaml
```

### 4.2 训练启动脚本

保存为 `train.sh`：

```bash
#!/bin/bash
set -e

echo "========================================="
echo "  SenseVoice Small 微调训练"
echo "========================================="

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sensevoice

# 检查数据
TRAIN_COUNT=$(wc -l < data/train.jsonl)
echo "训练数据: ${TRAIN_COUNT} 条"

if [ -f data/val.jsonl ]; then
    VAL_COUNT=$(wc -l < data/val.jsonl)
    echo "验证数据: ${VAL_COUNT} 条"
fi

# 检查 GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null || echo "警告: 未检测到 GPU"

# 启动训练
echo ""
echo "开始训练..."
python funasr/bin/train.py \
  --config-path conf \
  --config-name finetune.yaml \
  2>&1 | tee training_log.txt

echo ""
echo "✅ 训练完成！模型保存在 output/sensevoice_finetuned/"
```

```bash
chmod +x train.sh
./train.sh
```

### 4.3 训练过程解读

```
Epoch 1/20 ━━━━━━━━━━━━━━━━━━━━ 500/500 [0:05:23<0:00,  loss=2.341]
  train_loss: 2.341  |  val_loss: 1.892  |  lr: 0.000095

Epoch 2/20 ━━━━━━━━━━━━━━━━━━━━ 500/500 [0:05:18<0:00,  loss=1.756]
  train_loss: 1.756  |  val_loss: 1.523  |  lr: 0.000100

...

Epoch 10/20 ━━━━━━━━━━━━━━━━━━━ 500/500 [0:05:15<0:00,  loss=0.892]
  train_loss: 0.892  |  val_loss: 0.956  |  lr: 0.000050
```

| 指标 | 说明 |
|------|------|
| `train_loss` | 训练损失，持续下降=正常 |
| `val_loss` | 验证损失，降到不再降=收敛 |
| 过拟合信号 | train_loss 一直降，val_loss 开始上升 |

### 4.4 训练完成后的文件

```
output/sensevoice_finetuned/
├── best_model/             # 最佳模型（用于导出 ONNX）
│   ├── model.pt
│   ├── config.yaml
│   ├── tokens.txt
│   └── am.mvn
├── epoch_5/                # 中间 checkpoint
├── epoch_10/
├── epoch_15/
├── epoch_20/
└── training.log
```

---

## 五、评估模型

### 5.1 用测试集评估

```python
# evaluate.py
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import json

# 加载微调后的模型
model = AutoModel(
    model="./output/sensevoice_finetuned/best_model",
    trust_remote_code=True,
    device="cuda:0",
)

# 读取测试数据
test_file = "data/test.jsonl"
with open(test_file, "r", encoding="utf-8") as f:
    test_items = [json.loads(line) for line in f]

total = len(test_items)
correct = 0
errors = []

for item in test_items:
    result = model.generate(
        input=item["source"],
        language="zh",
        use_itn=True,
    )
    predicted = result[0]["text"].strip()
    expected = item["target"].strip()

    if predicted == expected:
        correct += 1
    else:
        errors.append({
            "key": item["key"],
            "expected": expected,
            "predicted": predicted,
        })

# 报告
print(f"\n===== 评估报告 =====")
print(f"总数: {total}")
print(f"完全匹配: {correct}")
print(f"准确率: {correct/total*100:.1f}%")
print(f"错误数: {len(errors)}")

if errors:
    print(f"\n错误样本（前 10 条）：")
    for e in errors[:10]:
        print(f"  期望: {e['expected']}")
        print(f"  预测: {e['predicted']}")
        print()
```

### 5.2 对比原始模型

```python
# compare_models.py
from funasr import AutoModel

test_audio = "data/audio_16k/sample_001.wav"

# 原始模型
print("=== 原始 SenseVoiceSmall ===")
model_base = AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True)
r1 = model_base.generate(input=test_audio, language="zh", use_itn=True)
print(r1[0]["text"])

# 微调模型
print("\n=== 微调后模型 ===")
model_ft = AutoModel(model="./output/sensevoice_finetuned/best_model", trust_remote_code=True)
r2 = model_ft.generate(input=test_audio, language="zh", use_itn=True)
print(r2[0]["text"])
```

---

## 六、常见问题

| 问题 | 解决方案 |
|------|----------|
| 显存不足 (CUDA OOM) | 减小 `batch_size`，增大 `accum_grad` |
| 报错找不到模型 | 确认 `model_dir` 路径正确，或首次运行让它自动下载 |
| 训练不收敛 | 降低 lr 到 5e-5，增加 warmup_steps |
| val_loss 不降反升 | 过拟合，增大数据量 / 增加 dropout / 冻结更多层 |
| 音频加载失败 | 检查文件格式：16kHz, 16bit, mono WAV |
| CUDA 不可用 | 检查 `nvidia-smi`，确认 CUDA Toolkit 版本匹配 |
| WSL2 开机后 GPU 丢失 | 管理员 PowerShell 执行 `wsl --shutdown`，重启 WSL |

---

## 七、下一步

训练完成后，接下来：

→ **[ONNX导出与香橙派部署教程.md](./ONNX导出与香橙派部署教程.md)**

---

> 📅 最后更新：2026-07-16  
> 🔗 参考：FunASR / SenseVoice / sherpa-onnx
