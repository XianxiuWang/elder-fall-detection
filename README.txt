================================================================================
  老人跌倒多模态AI监护系统 — 项目文件说明
  挑战杯揭榜挂帅 XH-202617
  最后更新: 2026-07-24
================================================================================

──────────────────────────────────────────────────────────────
  【2026-08-14 状态更新】
  - 生产模型 = v8 (51 维 XGBoost): models/fall_classifier_6class.pkl
    Fall 召回 97.1% / 误报 5.5/千窗 (EMA alpha=0.15, prob_thresh=0.40)
  - 活跃链路 = deploy_opi5/ (Orange Pi 5 板端自包含部署)
  - src/ 已归档: PC 端完整应用, 8-03 停更且 42 维加载器与 v8(51维)失配,
    详见 src/_ARCHIVED.md。若要恢复需把 ml_6class_detector.py 升级到 51 维提取器。
  - 以下 7-24 版本说明已过时 (仍描述 LightGBM 36维/Phase A), 仅供历史参考。
──────────────────────────────────────────────────────────────

【项目概述】
  基于 MediaPipe + LightGBM 的端到端跌倒检测系统。
  使用单目 RGB 摄像头（萤石 C6c）实时分析人体 33 关键点，
  结合空间运动特征 + 过程分析 + ML 分类器做跌倒判定。

  当前模型: 硬阈值启发式（已配置化） + ML 分类器（Phase A 开发中）
  目标平台: Windows 开发 / Orange Pi 5 Pro 部署
  环境: conda env fall (Python 3.8, mediapipe 0.10.14)

================================================================================
  一、文件夹说明
================================================================================

src/                         核心检测引擎（项目核心代码）
  │ 所有跌倒检测的核心逻辑都在这里。src/ 内部的文件互相引用，
  │ 使用相对 import（from .xxx import ...）。
  │
  ├── fall_config.py            集中式配置中心
  │    FallConfig 根配置 + 5 个子配置 (MotionSpatialConfig,
  │    ProcessFallConfig, GaitTrendConfig, E2EMonitorConfig,
  │    MediaPipeConfig)，共 107 个可调参数。
  │    支持 JSON 保存/加载，flat-dict 转换（用于网格搜索）。
  │    用法: from src.fall_config import FallConfig
  │
  ├── motion_spatial.py         空间运动分析器
  │    灵感来自 FMCW 雷达跌倒论文。每帧提取 6 维空间特征：
  │    运动中心位移、方向角、扩散范围、扩散宽度、躯干位移比、
  │    上下半身位移比。is_likely_fall() 为静态方法（无状态）。
  │    用法: from src.motion_spatial import MotionSpatialAnalyzer
  │
  ├── process_fall_detector.py  过程级跌倒检测器
  │    灵感来自"瞳芯颐护"的过程分析降低误报率思路。
  │    不是单帧判断，而是分析滑动窗口内的 5 维轨迹：
  │    速度曲线、高度变化、躯干角度、空间扩散、静止恢复。
  │    输出 FallAlert（5 级: STATIC/LOW/SUSPICIOUS/WARNING/ALERT/URGENT）。
  │    用法: from src.process_fall_detector import ProcessFallDetector
  │
  ├── gait_trend.py             步态趋势分析器
  │    长期（7天窗口）监测步态退化趋势，用于老年人状态预警。
  │    分析指标: 步长、步行速度、摇摆角度、平衡指数等 7 个维度。
  │    用法: from src.gait_trend import GaitTrendAnalyzer
  │
  ├── ml_fall_detector.py       ML 推理模块
  │    加载训练好的 LightGBM 模型，对实时关键点流做滑动窗口推理。
  │    窗口=30帧，每5帧推理一次，内置防抖机制（连续3次阳性才触发）。
  │    用法: from src.ml_fall_detector import MLFallDetector
  │          detector = MLFallDetector("models/fall_classifier.pkl")
  │          prob, label, is_fall = detector.update(landmarks)
  │
  └── e2e_fall_monitor.py       端到端监控主入口
  │    整合上述所有模块，提供完整的实时跌倒监控界面。
  │    支持 RTSP 摄像头 + 本地摄像头回退。
  │    用法: python -m src.e2e_fall_monitor
  │

training/                    模型训练 & 参数调优
  │
  ├── train_fall_classifier.py  跌倒 ML 分类器训练脚本
  │    从模拟数据或 URFD 特征训练 LightGBM/XGBoost/RF 模型。
  │    流程: 生成/加载数据 → 滑动窗口特征提取(36维) →
  │          交叉验证 → 训练 → 评估 → 保存模型。
  │    用法: python training/train_fall_classifier.py --synthetic
  │          python training/train_fall_classifier.py --compare  (对比3种模型)
  │
  └── param_tuner.py            参数自动调优脚本（Phase C）
  │    6 类模拟场景（行走/跌倒/弯腰/坐下/挥手/蹲下），每类 15 变体。
  │    随机搜索 4000 组参数 + Top-20 精炼，输出最优配置。
  │    结果: baseline 0.8718 → best 1.0000 (+14.7%)
  │    用法: python training/param_tuner.py --search 4000
  │

data_tools/                   数据采集 & 加载工具
  │
  ├── data_collector.py         跌倒数据采集录制工具
  │    配合摄像头录制标注视频片段。
  │    操作: SPACE=录制, 1-9=选动作标签, R=看进度, Q=退出。
  │    支持 RTSP + 本地摄像头，录制时实时显示标签和计时。
  │    用法: python data_tools/data_collector.py
  │          python data_tools/data_collector.py --rtsp "rtsp://..."
  │
  └── urfd_loader.py            URFD 数据集加载器
  │    将 URFD 公开数据集的 PNG 序列 → MediaPipe 33关键点 →
  │    空间运动特征 → 保存为 .npz 格式供训练使用。
  │    输入: URFD 原始目录 (Fall/ + ADL/ 子文件夹)
  │    输出: 每序列一个 .npz + dataset_summary.json
  │    用法: python data_tools/urfd_loader.py --data_dir E:/datasets/URFD
  │

utils/                        工具脚本
  │
  ├── test_camera.py            摄像头连接测试
  │    快速测试摄像头是否可用，显示实时画面。
  │    用法: python utils/test_camera.py
  │
  └── download_pose_model.py    MediaPipe 姿态模型下载器
  │    下载 pose_landmarker.task 到本地。
  │    用法: python utils/download_pose_model.py
  │

models/                       模型文件 & 配置参数
  │
  ├── pose_landmarker.task      MediaPipe 姿态估计模型文件
  │    用于提取人体 33 个关键点坐标。
  │
  ├── params_best.json          参数调优最佳结果（Phase C 产出）
  │    包含 107 个参数的最优组合，得分 1.0000。
  │    调用 fall_config.py 的 load_or_default() 自动加载。
  │
  └── params_default.json       参数调优基准线（baseline 0.8718）
  │    优化前的默认参数配置，用于对比。
  │

logs/                         运行日志
  │
  └── tuner_log.txt             参数调优完整日志
  │    记录 4000 次随机搜索 + 600 次精炼的进度和结果。
  │

docs/                         文档
  │
  └── 刷机教程.md               Orange Pi 5 Pro 系统烧录教程
  │    Joshua-Riek Ubuntu 24.04 v2.3.2 镜像刷写步骤。
  │

deploy/                       部署相关
  │
  └── orangepi5pro-ubuntu-24.04-server-v2.3.2.img.xz
  │    Orange Pi 5 Pro 系统镜像（~4GB），已烧录到 TF 卡。
  │

================================================================================
  二、常用命令速查
================================================================================

【实时监控】
  conda activate fall
  python -m src.e2e_fall_monitor

【录制标注数据】
  conda activate fall
  python data_tools/data_collector.py

【训练 ML 模型（模拟数据）】
  conda activate fall
  python training/train_fall_classifier.py --synthetic

【训练 ML 模型（URFD 真实数据）】
  conda activate fall
  python data_tools/urfd_loader.py --data_dir E:/datasets/URFD
  python training/train_fall_classifier.py --data_dir E:/老人跌倒/data/urfd_features/

【多模型对比】
  conda activate fall
  python training/train_fall_classifier.py --synthetic --compare

【参数调优】
  conda activate fall
  python training/param_tuner.py --search 4000

【测试 ML 推理器】
  conda activate fall
  python -c "from src.ml_fall_detector import MLFallDetector; MLFallDetector('models/fall_classifier.pkl')"

================================================================================
  三、数据流
================================================================================

  [摄像头/视频] ──→ [MediaPipe] ──→ [33关键点/帧]
                                        │
                     ┌──────────────────┼──────────────────┐
                     ▼                  ▼                  ▼
              motion_spatial    process_fall         gait_trend
              (每帧6维特征)     (窗口5维分析)      (长期步态趋势)
                     │                  │                  │
                     └────────┬─────────┘                  │
                              ▼                            │
                        ML 分类器 ◄─────────────────────────┘
                      (LightGBM, 36维)
                              │
                              ▼
                       双重判定输出
                  (SUSPICIOUS/WARNING/ALERT/URGENT)

================================================================================
  四、开发阶段
================================================================================

  Phase D ✅  修 Bug + 配置化重构           (完成: 2026-07-24)
  Phase C ✅  参数自动调优                   (完成: 2026-07-24)
  Phase A 🔄  ML 分类器训练                  (进行中: 待安装 sklearn/lightgbm)
  Phase B ⏳  LSTM 时序建模                   (待规划)
  部署   ⏳  Orange Pi 5 Pro 端侧推理         (用户暂缓)

================================================================================
  五、重要环境参数
================================================================================

  摄像头:  萤石 C6c, IP 192.168.1.100:554
  RTSP:    rtsp://admin:RVXCEM@192.168.1.100:554/h264/ch1/main/av_stream
  验证码:  RVXCEM
  注意:   需重新配网到当前环境（192.168.37.x）

  开发机:  Windows 10, Python 3.8 (conda env fall)
  目标机:  Orange Pi 5 Pro, Ubuntu 24.04, 16GB RAM

  conda 环境:
    fall     ─ 跌倒检测 (mediapipe 0.10.14 + opencv)
    sse      ─ SenseVoice 语音训练

================================================================================
  六、关键数据集
================================================================================

  URFD (UR Fall Detection)    ─ 30跌倒+40日常, RGB+深度, Kaggle可下载
  UP-Fall                     ─ 11人×11动作, 多模态
  Le2i Fall Detection         ─ 191段视频, 多场景
  自采集                      ─ 计划36段(13跌倒+23日常), 使用data_collector.py

================================================================================
