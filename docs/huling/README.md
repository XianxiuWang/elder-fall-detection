# 护龄 — 状态分类模型训练工具

## 项目结构

```
huling_model/
├── config.py            # 全局配置（参数、路径、状态定义）
├── collect_data.py      # 摄像头采集 + MediaPipe 提取关键点 → 保存为数据集
├── extract_features.py  # 特征工程：33个关键点 → 115维特征向量
├── train.py             # 训练流水线（K折交叉验证 + 多模型对比 + 调参）
├── inference_demo.py    # 实时推理演示（摄像头 + 骨架 + 分类结果）
├── export_model.py      # 模型导出（ONNX / TFLite / pickle）
└── models/              # 保存的训练模型
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 采集你自己的数据（和同学一起录）
python collect_data.py

# 3. 训练模型
python train.py

# 4. 实时推理演示
python inference_demo.py

# 5. 导出模型（给嵌入式端用）
python export_model.py
```

## 7类状态定义

| ID | 状态 | 说明 |
|----|------|------|
| 0 | 正常行走 | walking |
| 1 | 坐着/休息 | sitting |
| 2 | 躺卧（正常）| lying |
| 3 | 久坐未动 | sedentary |
| 4 | 异常姿态 | abnormal |
| 5 | 跌倒/倒地 | fall |
| 6 | 无人活动 | empty |
