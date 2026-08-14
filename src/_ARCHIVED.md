# src/ 目录 — 已归档（2026-08-14）

## 这是什么

PC 端完整"养老陪伴"应用（`python -m src.e2e_fall_monitor`），功能远多于板端精简版：
多人检测(YOLOv8n+IoU)、踉跄前兆状态机、行为画像、个性化步态基线、久坐检测、
服药提醒、暖关怀语音等。

## 为什么归档（不删除）

1. **停更**：src/ 内所有文件最后修改时间为 2026-08-03，之后维护重心转向板端 `deploy_opi5/`。
2. **已失配（跑不通）**：`ml_6class_detector.py` 用的是 42 维 base 提取器
   （`train_fall_classifier.FeatureExtractor`），但主模型 `models/fall_classifier_6class.pkl`
   自 8-13 起为 59 维 v6（现为 51 维 v8）。实测加载后第一次推理报错：
   ```
   ValueError: X has 42 features, but StandardScaler is expecting 59 features
   ```
3. **价值仍在**：行为画像 / 步态趋势 / 踉跄前兆 / 服药提醒 / 语音关怀是答辩/演示的
   核心卖点，比板端精简版更能体现工作量。故保留归档，暂不删除。

## 若需恢复 / 迁移

把 `src/ml_6class_detector.py` 的加载器从 42 维升级到 51 维 v8 提取器：

```python
# 旧（42 维，已失配）
from train_fall_classifier import FeatureExtractor
self.extractor = FeatureExtractor(window_size=window_size)

# 新（51 维，对齐 v8）
from deploy_opi5.fall_inference import EnhancedFeatureExtractor   # 或 training 内等价实现
self.extractor = EnhancedFeatureExtractor(window_size=window_size)
```

其余逻辑（防抖、置信度累积、多数表决）无需改动。预计工作量 0.5~1 小时。

## 当前活跃链路

- **板端**：`deploy_opi5/`（自包含，8-13/8-14 更新，v8 模型 + alpha=0.15 / thresh=0.40）
- **模型**：`models/fall_classifier_6class.pkl` = v8（51 维），与板端统一
- 版本存档：v6/v7/v8 各自保留在 `models/fall_classifier_6class_v{6,7,8}.pkl`
