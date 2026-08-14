# ONNX 导出与量化教程

> 接续：SenseVoice 微调训练完成  
> 用途：把微调好的 PyTorch 模型导出为 ONNX → int8 量化 → 为部署做准备

---

## 一、前提条件

确保微调训练已完成，且有以下目录：

```
SenseVoice/
└── output/
    └── sensevoice_finetuned/
        └── best_model/          # 最佳 checkpoint
            ├── model.pt
            ├── config.yaml
            ├── tokens.txt
            └── am.mvn
```

---

## 二、克隆 sherpa-onnx 导出脚本

```bash
cd ~
git clone https://github.com/k2-fsa/sherpa-onnx.git
cd sherpa-onnx/scripts/sense-voice/

# 安装导出所需依赖
pip install onnx onnxruntime onnx-simplifier onnxoptimizer
```

> 📦 国内加速：`git clone https://gitee.com/mirrors/sherpa-onnx.git`

---

## 三、导出 ONNX

### 3.1 方法 A：修改官方脚本运行

`sherpa-onnx/scripts/sense-voice/` 下有两个关键文件：

| 文件 | 作用 |
|------|------|
| `export-onnx.py` | 核心导出逻辑 |
| `run.sh` | 一键运行脚本 |

**修改 `run.sh`**，把模型路径指向你微调好的模型：

```bash
#!/bin/bash
# run.sh - 导出微调后的 SenseVoice 模型

# 微调后的模型路径
SENSEVOICE_MODEL_DIR=~/SenseVoice/output/sensevoice_finetuned/best_model

# 输出目录
OUTPUT_DIR=~/sensevoice_exported
mkdir -p $OUTPUT_DIR

cd $(dirname $0)

python3 export-onnx.py \
  --model-dir $SENSEVOICE_MODEL_DIR \
  --output-dir $OUTPUT_DIR

echo "✅ ONNX 模型已导出到: $OUTPUT_DIR"
```

运行：

```bash
cd ~/sherpa-onnx/scripts/sense-voice/
bash run.sh
```

### 3.2 方法 B：直接用 Python 导出

如果官方脚本有兼容性问题，用下面的独立脚本：

```python
# export_my_sensevoice.py
"""
将微调后的 SenseVoice PyTorch 模型导出为 ONNX
"""
import os
import torch
import onnx
import json

# ===== 配置 =====
MODEL_DIR = os.path.expanduser("~/SenseVoice/output/sensevoice_finetuned/best_model")
OUTPUT_DIR = os.path.expanduser("~/sensevoice_exported")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 加载微调后的模型 =====
print(f"加载模型: {MODEL_DIR}")
from funasr import AutoModel

model = AutoModel(
    model=MODEL_DIR,
    trust_remote_code=True,
    device="cpu",  # 导出时用 CPU
)

# ===== 导出为 ONNX =====
print("正在导出 ONNX...")

# 构造一个示例输入（16kHz 音频，1 秒 = 16000 个采样点）
dummy_input = torch.randn(1, 16000 * 5)  # 5 秒音频

output_path = os.path.join(OUTPUT_DIR, "model.onnx")

torch.onnx.export(
    model.model,                    # 底层 PyTorch 模型
    dummy_input,                    # 示例输入
    output_path,                    # 输出路径
    input_names=["speech"],
    output_names=["text", "timestamps"],
    dynamic_axes={
        "speech": {0: "batch", 1: "samples"},  # 支持可变长度
    },
    opset_version=15,
    do_constant_folding=True,
)

print(f"✅ float32 ONNX: {output_path}")

# 验证 ONNX 模型
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("✅ ONNX 模型验证通过")

# 复制 tokens.txt
import shutil
src_tokens = os.path.join(MODEL_DIR, "tokens.txt")
dst_tokens = os.path.join(OUTPUT_DIR, "tokens.txt")
if os.path.exists(src_tokens):
    shutil.copy(src_tokens, dst_tokens)
    print(f"✅ tokens.txt 已复制")

print(f"\n导出完成！文件位于: {OUTPUT_DIR}")
```

运行：

```bash
python export_my_sensevoice.py
```

### 3.3 导出产物

成功后得到：

```
~/sensevoice_exported/
├── model.onnx          # float32，~894MB
├── model.int8.onnx     # int8 量化（如果 run.sh 自动生成了）
└── tokens.txt          # 词表
```

---

## 四、int8 量化

float32 模型 894MB 太大，香橙派 4GB 内存跑不动，必须量化。

### 4.1 动态量化（简单，精度损失小）

```bash
pip install onnxruntime

python -c "
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

input_model = './model.onnx'
output_model = './model.int8.onnx'

quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QUInt8,    # 权重量化为 int8
    extra_options={},
)

import os
size_before = os.path.getsize(input_model) / 1024 / 1024
size_after = os.path.getsize(output_model) / 1024 / 1024
print(f'✅ 量化完成: {size_before:.0f}MB → {size_after:.0f}MB (减少 {size_before-size_after:.0f}MB)')
"
```

### 4.2 静态量化（更精准，需要校准数据）

```python
# static_quantize.py
import onnx
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
import numpy as np
import soundfile as sf

class SenseVoiceCalibrationDataReader(CalibrationDataReader):
    """用你的训练数据做校准"""
    def __init__(self, audio_files, sample_rate=16000):
        self.audio_files = audio_files
        self.sample_rate = sample_rate
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.audio_files):
            return None

        audio, sr = sf.read(self.audio_files[self.idx], dtype='float32')
        if sr != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

        # 确保是单声道
        if audio.ndim > 1:
            audio = audio[:, 0]

        self.idx += 1
        return {"speech": audio.reshape(1, -1).astype(np.float32)}

# 收集校准数据
import glob
audio_files = glob.glob("data/audio_16k/*.wav")[:100]  # 取 100 条做校准就够了
reader = SenseVoiceCalibrationDataReader(audio_files)

quantize_static(
    model_input="./model.onnx",
    model_output="./model.int8.onnx",
    calibration_data_reader=reader,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QUInt8,
)
print("✅ 静态量化完成")
```

### 4.3 验证量化模型

```bash
# 用 onnxruntime 跑一次推理，对比量化前后
python -c "
import onnxruntime as ort
import numpy as np
import soundfile as sf
import time

# 加载音频
audio, sr = sf.read('data/audio_16k/sample_001.wav', dtype='float32')
audio = audio.reshape(1, -1)

# float32 推理
sess_fp32 = ort.InferenceSession('./model.onnx')
t0 = time.time()
out_fp32 = sess_fp32.run(None, {'speech': audio})
t_fp32 = time.time() - t0

# int8 推理
sess_int8 = ort.InferenceSession('./model.int8.onnx')
t0 = time.time()
out_int8 = sess_int8.run(None, {'speech': audio})
t_int8 = time.time() - t0

print(f'float32 耗时: {t_fp32:.2f}s')
print(f'int8    耗时: {t_int8:.2f}s')
print(f'加速比:    {t_fp32/t_int8:.1f}x')
"
```

### 4.4 量化效果参考

| 模型 | 大小 | 推理时间（5秒音频） |
|------|------|-------------------|
| float32 | ~894MB | 1.2s (CPU) |
| int8 动态量化 | ~228MB | 0.6s (CPU) |
| int8 静态量化 | ~228MB | 0.5s (CPU) |

**结论：int8 量化减 75% 体积 + 翻倍速度，精度几乎无损。**

---

## 五、打包待部署

```bash
# 创建部署包
mkdir -p ~/sensevoice_deploy/
cp ~/sensevoice_exported/model.int8.onnx ~/sensevoice_deploy/
cp ~/sensevoice_exported/tokens.txt ~/sensevoice_deploy/

# 打包
cd ~
tar czf sensevoice_deploy.tar.gz sensevoice_deploy/
ls -lh sensevoice_deploy.tar.gz
# 约 230MB，准备传到香橙派
```

---

## 六、常见问题

| 问题 | 解决 |
|------|------|
| `ImportError: funasr` | `pip install funasr` |
| 导出脚本报错 | 检查 `model_dir` 路径是否正确 |
| ONNX 模型超过 2GB | 正常，后续量化会缩小到 230MB |
| 量化后精度大幅下降 | 改用静态量化 + 用自己的数据做校准 |
| 导出时 CPU 内存不足 | 关闭其他程序，或用 `device="cpu"` 分批处理 |
| tokens.txt 路径问题 | 手动复制 `cp ../best_model/tokens.txt ./` |

---

## 七、下一步

ONNX 模型已就绪，接下来：

→ **[香橙派 5 Pro 部署教程.md](./香橙派5Pro部署教程.md)**

---

> 📅 最后更新：2026-07-16  
> 🔗 参考：sherpa-onnx / onnxruntime
