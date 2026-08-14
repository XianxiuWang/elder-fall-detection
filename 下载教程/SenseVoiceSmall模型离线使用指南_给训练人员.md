# SenseVoiceSmall 模型离线使用指南

> ⚠️ 本文档配合压缩包 `sensevoice_model.tar.gz` 使用  
> 用途：离线安装 SenseVoiceSmall 模型 + 加载验证 + 准备微调

---

## 一、前置要求

| 需求 | 说明 |
|------|------|
| 操作系统 | Windows / Linux 均可 |
| Python | 3.10（推荐） |
| GPU | NVIDIA 显卡（6GB+ 显存），训练必选 |
| CUDA | 12.1（随 PyTorch 自动安装） |
| 磁盘空间 | 至少 5GB（模型 900MB + PyTorch 约 2GB + 依赖） |

---

## 二、接收并解压模型

### 2.1 将压缩包放到 E 盘

把 `sensevoice_model.tar.gz`（约 850MB）放到 `E:\` 根目录。

### 2.2 解压

```cmd
cd /d E:\
tar -xzf sensevoice_model.tar.gz
```

如果 `tar` 不行，用 7-Zip：右键 → 7-Zip → 提取到当前位置。

### 2.3 确认解压成功

```cmd
dir E:\iic--SenseVoiceSmall\snapshots\master
```

应该看到：

```
am.mvn
chn_jpn_yue_eng_ko_spectok.bpe.model
config.yaml
configuration.json
model.pt               ← 核心权重文件，936MB
tokens.json
README.md
example/               ← 示例脚本
fig/
```

---

## 三、安装运行环境

### 3.1 创建 conda 环境

```cmd
conda create -n sensevoice python=3.10 -y
conda activate sensevoice
```

### 3.2 安装 PyTorch（GPU 版）

```cmd
conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

验证：

```cmd
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

输出 `True NVIDIA GeForce RTX xxxx` 即正常。

> 如果没有 NVIDIA 显卡，只做推理不做训练：
> ```cmd
> conda install pytorch torchaudio cpuonly -c pytorch -y
> ```

### 3.3 安装训练框架

```cmd
pip install funasr
```

---

## 四、加载模型验证

```cmd
python -c "
from funasr import AutoModel

model = AutoModel(
    model='E:/iic--SenseVoiceSmall/snapshots/master',
    trust_remote_code=True,
    device='cuda:0',
)

print('✅ 模型加载成功')
print(f'设备: {model.device}')

# 用模型自带的示例音频测试
import os
example_dir = 'E:/iic--SenseVoiceSmall/snapshots/master/example'
wavs = [f for f in os.listdir(example_dir) if f.endswith('.wav')]
if wavs:
    result = model.generate(
        input=os.path.join(example_dir, wavs[0]),
        language='zh',
        use_itn=True,
    )
    print(f'测试结果: {result}')
else:
    print('没有示例音频，跳过测试')
"
```

看到 `✅ 模型加载成功` 就说明一切正常。

---

## 五、准备训练数据

### 5.1 数据目录结构

```
E:\SenseVoice_training\
├── data\
│   ├── audio\              # 16kHz WAV 音频
│   │   ├── sample_001.wav
│   │   ├── sample_002.wav
│   │   └── ...
│   ├── train.jsonl         # 训练集
│   └── val.jsonl           # 验证集
│
└── conf\
    └── finetune.yaml       # 训练配置
```

### 5.2 训练数据格式（train.jsonl）

每行一个 JSON：

```jsonl
{"key": "sample_001", "source": "data/audio/sample_001.wav", "target": "你好我是市公安局的你涉嫌洗钱请配合调查", "source_lang": "zh"}
{"key": "sample_002", "source": "data/audio/sample_002.wav", "target": "您的医保卡在上海被人盗刷了需要核实信息", "source_lang": "zh"}
{"key": "sample_003", "source": "data/audio/sample_003.wav", "target": "今天天气真好我们出去走走吧", "source_lang": "zh"}
```

| 字段 | 说明 |
|------|------|
| `key` | 唯一 ID |
| `source` | 音频文件路径 |
| `target` | 标注文本 |
| `source_lang` | zh / en / ja / ko / yue |

### 5.3 音频格式要求

- WAV 格式
- 16kHz 采样率
- 16bit 位深度
- 单声道
- 每条 2~20 秒

---

## 六、开始微调训练

### 6.1 训练配置（conf/finetune.yaml）

```yaml
model: SenseVoiceSmall
model_dir: E:/iic--SenseVoiceSmall/snapshots/master

train_data: data/train.jsonl
valid_data: data/val.jsonl

batch_size: 4
accum_grad: 4
max_epoch: 20
learning_rate: 0.0001
warmup_steps: 500

freeze_frontend: true
freeze_encoder: false

output_dir: ./output/sensevoice_finetuned
save_checkpoint_interval: 1000
log_interval: 50

seed: 42
use_amp: true
```

### 6.2 启动训练

```cmd
cd /d E:\SenseVoice_training
conda activate sensevoice

python -c "
from funasr.bin import train
train.main([
    '--config-path', 'conf',
    '--config-name', 'finetune.yaml',
])
"
```

### 6.3 训练完成后的输出

```
output/sensevoice_finetuned/
├── best_model/          # 最佳 checkpoint ← 用于后续导出
│   ├── model.pt
│   ├── config.yaml
│   └── tokens.txt
├── epoch_5/
├── epoch_10/
├── ...
└── training.log
```

---

## 七、快速检查清单

| 步骤 | 命令 |
|------|------|
| 1. 解压模型 | `tar -xzf sensevoice_model.tar.gz` |
| 2. 创建环境 | `conda create -n sensevoice python=3.10 -y` |
| 3. 装 PyTorch | `conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y` |
| 4. 装 FunASR | `pip install funasr` |
| 5. 加载验证 | `AutoModel(model='E:/iic--SenseVoiceSmall/snapshots/master')` |
| 6. 准备数据 | 见第五章 |
| 7. 开始训练 | 见第六章 |

---

## 八、常见问题

| 问题 | 解决 |
|------|------|
| 解压后找不到 model.pt | 检查压缩包是否完整（约 850MB），重新传输 |
| `AutoModel` 报路径错误 | 路径用正斜杠：`E:/iic--SenseVoiceSmall/snapshots/master` |
| CUDA 不可用 | 检查 `nvidia-smi`，确认驱动程序已安装 |
| 显存不足 (OOM) | 减小 `batch_size` 到 2，增大 `accum_grad` 到 8 |
| 训练不收敛 | 降低 learning_rate 到 5e-5，检查数据格式 |

---

> 📅 版本：SenseVoiceSmall (iic/SenseVoiceSmall)  
> 🔧 框架：FunASR + PyTorch 2.x  
> 📦 模型大小：936MB (model.pt)
