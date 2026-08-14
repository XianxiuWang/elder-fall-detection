# HANDOFF.md — 挑战杯赛题 XH-202617 项目全貌

**最后更新**: 2026-07-18
**写给**: 完全没有上下文的新会话
**目的**: 让任何新会话打开这个文件就能立刻知道项目全貌、当前进度、卡在哪、下一步做什么

---

## 一、项目是什么

**比赛**: "挑战杯"揭榜挂帅专项赛 — 海康威视/萤石发榜
**赛题编号**: XH-202617
**赛题名称**: 基于多模态 AI 监测的老年人跌倒风险、心理健康、诈骗识别及预警研究
**项目名**: AI 智能居家监护系统
**核心定位**: 从"被动报警"升级为"主动预测"，从"冷监控"升级为"暖陪伴"

**三大创新模块**:
1. **跌倒风险预测** — MediaPipe 33 骨骼点 → 步态分析 → 提前 3-7 天预警
2. **心理健康趋势追踪** — 表情识别 + 语音情绪 + 活动量 → 情绪曲线
3. **LLM 赋能交互** — 无感认知评估、回忆疗法引擎、家人声音克隆

**硬件平台**: Orange Pi 5 Pro（RK3588S, 4GB LPDDR5, 32GB eMMC, 6 TOPS NPU）
**摄像头**: 萤石 C6c 室内云台 ×1 + USB/普通摄像头(门口) ×1
**部署方式**: 全边缘计算，视频不出本地（隐私优先）
**截止日期**: 2026年9月5日提交，8周开发计划

---

## 二、团队分工

| # | 角色 | 姓名 | 职责 |
|---|------|------|------|
| 1 | 项目负责人 | 钱永佳 | 架构设计 + 进度管理 + 答辩材料 |
| 2 | CV全栈+边缘一体化 | **王纤秀(你)** | 模型训练→RKNN部署全包 + 萤石对接 + 创新架构 |
| 3 | CV人脸+表情 | 王权瑶 | 人脸/情绪 + 最终项目书撰写 |
| 4 | NLP/语音 | （待定） | 语音转写 + 诈骗检测 + LLM 对话引擎 |
| 5 | 后端 | （待定） | FastAPI + 数据库 + LLM API 集成 |
| 6 | 前端 | （待定） | Vue.js + 健康仪表盘 + 告警推送 |
| 7 | 测试与数据 | （待定） | 数据采集标注 + 全链路测试 + 演示环境 |

---

## 三、已经完成的工作

### 3.1 文档产出（E:\老人跌倒\ 目录下）

| 文件/目录 | 说明 |
|-----------|------|
| `萤石开放平台资源使用说明.docx` | 详细版，9项能力 + 架构关系 |
| `萤石开放平台资源申请清单.docx` | 简洁版，4张表一页搞定 |
| `项目计划实施书.md` + `.docx` | V2.3 版，8周计划，硬件参数已更新 |
| `系统架构_mermaid.md` | Mermaid 格式架构图，可粘贴到飞书渲染 |
| `方案设计PPT提示词.txt` | 5 个模块的 PPT 生成提示词，喂给 Gamma.app |
| `npu-setup/` | 4 个 NPU 脚本（README + verify + convert + infer） |
| `下载教程/` | 5 份 SenseVoice 教程文档 |
| `HANDOFF.md` | 本文件 |
| `本轮对话总结.md` | 最新一轮对话概要 |

### 3.2 PPT 提示词（已生成，可直接用）

位置: `E:\老人跌倒\方案设计PPT提示词.txt`

包含 5 个模块：
- **模块一**: 市场调研与需求分析（4-5页）— 痛点、政策、用户画像、市场价值
- **模块二**: 技术架构与实现方案（6-7页）— 架构图、三大创新、技术栈、开发计划
- **模块三**: 硬件设备需求说明（2-3页）
- **模块四**: 萤石开放平台资源申请（3-4页）
- **模块五**: 预期成果与落地可行性（3-4页）
- 附赠：封面页、目录页、团队介绍页、致谢页

**使用方法**: 去 gamma.app（在线网页）→ Create new → Paste in text → 逐个模块粘贴生成

### 3.3 SenseVoice 环境（PC端）

**已完成**:
- ✅ Conda 环境 `ssv`（Python 3.10）创建成功
- ✅ modelscope 安装成功
- ✅ SenseVoiceSmall 预训练模型下载成功（`E:\SenseVoice_offline\models\iic--SenseVoiceSmall\snapshots\master\`）
- ✅ 模型打包为 `E:\sensevoice_model.tar.gz`（≈850MB），可发给训练人员
- ✅ 训练人员离线使用指南已生成（`E:\老人跌倒\下载教程\SenseVoiceSmall模型离线使用指南_给训练人员.md`）

**待验证**: `tar` 打包是否真的成功（命令无输出即为成功，可 `dir E:\sensevoice_model.tar.gz` 确认）

### 3.4 诈骗话术数据集渠道梳理

已整理 10 个平台 + 自建方案，详见 `E:\老人跌倒\交接文档\本轮对话总结.md` 第一章。

---

## 四、当前卡在哪里

### ⛔ 阻塞项 1：硬件未到货
- Orange Pi 5 Pro — **没有**
- 萤石 C6c 摄像头 — **没有**
- 所有 NPU 脚本、SDK 对接都依赖硬件

### ⛔ 阻塞项 2：硬件申请已错过
- 比赛免费硬件申请时间窗已过
- 需要**自购**: 萤石 C6c 600万 ≈ ¥229（推荐方案）

### ⛔ 阻塞项 3：专家还没联系
- 已起草电话提纲和邮件草稿
- 需要联系程老师/李老师确认：SDK对接、平台能力、硬件补救

### ⛔ 阻塞项 4：SenseVoice 还没开始训练
- PC 上还需要装 PyTorch（ssv 环境里还没装）
- 训练人员那边需要收到 `sensevoice_model.tar.gz` 才能开始

---

## 五、下一步计划（按优先级）

### P0 — 立刻做
1. **确认打包成功**: `dir E:\sensevoice_model.tar.gz`，确认文件存在
2. **联系专家**: 先打电话、后补邮件，问清楚硬件补救 + SDK对接 + 平台能力清单
3. **自购萤石 C6c**: 京东/淘宝下单 C6c 600万（¥229），不要再等

### P1 — 本周做
4. **发送模型给训练人员**: 网盘/QQ传 `sensevoice_model.tar.gz` + 离线使用指南
5. **PC 上装 PyTorch**: `conda activate ssv` → `conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y`
6. **SenseVoice 微调环境验证**: 确保 `AutoModel` 能加载本地模型

### P2 — 下周做
7. **Orange Pi 5 Pro 到手后**: 刷系统 → 运行 `verify_npu.py` → 装 sherpa-onnx
8. **萤石 C6c 到手后**: 注册开放平台 → 绑定设备 → 跑通 EZVIZ SDK 取流
9. **诈骗音频数据准备**: B站反诈视频下载 + TTS 批量生成 + 标注

### P3
10. SenseVoice 微调 → ONNX 导出 → int8 量化 → 部署到香橙派
11. 联调全链路：摄像头取流 → 语音识别 → 诈骗检测 → 告警推送

---

## 六、踩过的坑（绝对不要再踩）

### 坑 1: Python 版本不兼容
- **症状**: `from modelscope import snapshot_download` 在 Python 3.8 报错
- **根因**: modelscope 新版本需要 Python 3.9+
- **解决**: 创建 conda 环境时指定 `python=3.10`
- **教训**: 所有新项目都用 Python 3.10+，不要用旧环境

### 坑 2: pip 版本太老
- **症状**: `--no-proxy` 参数不被识别
- **解决**: `pip install modelscope`（不用额外参数）
- **也可以**: `pip install --upgrade pip` 升级 pip

### 坑 3: 系统代理干扰所有网络请求
- **症状**: pip/conda 莫名其妙 SSL 错误、连接超时
- **解决**: 每次开新 CMD 先跑：
  ```cmd
  set HTTP_PROXY=
  set HTTPS_PROXY=
  set http_proxy=
  set https_proxy=
  ```
- **教训**: 如果装了代理工具（Clash/V2Ray），关掉或设好规则

###坑 4: Conda 国内镜像全挂
- **症状**: `.condarc` 配了一堆清华/阿里镜像，全返回 404
- **解决**: `del %USERPROFILE%\.condarc` 删掉配置，用默认源
- **教训**: 当镜像不稳定时，默认源反而更可靠；别在 .condarc 里堆一堆无效镜像

### 坑 5: tar 命令在 Windows 上无输出
- **症状**: `tar -czf ...` 跑完没有任何输出
- **结论**: 没输出 = 成功（Windows tar 和 Linux 一样，没报错就是成了）
- **验证**: `dir E:\sensevoice_model.tar.gz` 看文件大小

### 坑 6: Git clone 代理不稳定
- **症状**: `ghproxy.com`、`fastgit.xyz` 等代理时好时坏
- **替代方案**: 浏览器直接下载 GitHub ZIP 包，解压后手动改名
- **已装 Git 的话**: `winget install --id Git.Git -e --source winget` 一键安装

---

## 七、关键文件清单

### 项目文档 (E:\老人跌倒\)

```
老人跌倒/
├── HANDOFF.md                              ← 本文件（每次会话更新）
├── 本轮对话总结.md
├── 萤石开放平台资源使用说明.docx
├── 萤石开放平台资源申请清单.docx
├── 010-赛题说明/                           ← 比赛方案 PDF 等原始材料
├── 比赛方案/
│   ├── 项目计划实施书.md / .docx           ← 8周 V2.3版
│   ├── 系统架构_mermaid.md
│   └── 方案设计PPT提示词.txt
├── npu-setup/
│   ├── README.md
│   ├── verify_npu.py                       ← 板子端一键检测
│   ├── convert.py                          ← PC端 ONNX→RKNN 转换
│   └── infer.py                            ← 板子端 NPU 推理测试
├── 下载教程/
│   ├── SenseVoice模型下载教程.md
│   ├── SenseVoice微调训练教程.md
│   ├── ONNX导出与量化教程.md
│   ├── 香橙派5Pro部署教程.md
│   └── SenseVoiceSmall模型离线使用指南_给训练人员.md
├── fraud_data_crawler/                     ← 诈骗数据爬虫（独立子项目）
└── 参考方案/                               ← 往届获奖作品参考
```

### SenseVoice 相关 (E:\)

```
E:\
├── SenseVoice\                             ← FunAudioLLM/SenseVoice 源码
├── SenseVoice_offline\                     ← modelscope 下载的模型
│   └── models\iic--SenseVoiceSmall\snapshots\master\
│       ├── model.pt                        (PyTorch 权重, ~900MB)
│       ├── config.yaml
│       ├── tokens.txt
│       └── ...
└── sensevoice_model.tar.gz                 ← 打包好的模型 (~850MB)，发给训练人员用
```

### Conda 环境

```
环境名: ssv
Python: 3.10
已装: modelscope
待装: pytorch, torchaudio, funasr
```

---

## 八、架构决策记录

| 决策 | 理由 |
|------|------|
| 萤石=感知层, 自研AI=分析层 | 充分依托平台已有能力（跌倒检测/人脸识别），在上面叠加创新 |
| Orange Pi 5 Pro 而非 Jetson | 比赛方案已基于 RK3588，预算有限；RKNN 工具链代替 TensorRT |
| SenseVoice 而非 Whisper | 中文识别碾压级优势 + RK3588 NPU 加速支持 |
| Web前端而非 App | 降低开发成本，家属用浏览器即可，兼容性更好 |
| 全边缘计算 | 隐私保护是差异化卖点，视频不出本地 |

---

## 九、未解决/待确认

| 问题 | 优先级 |
|------|:--:|
| 萤石 SDK 在 ARM64 (RK3588S) 上是否可用？文档/Demo？ | 🔴 高 |
| 萤石平台是否有云端 AI 能力（语音识别、大模型）可以互补边缘算力？ | 🔴 高 |
| 4GB 内存是否够同时跑多路 CV + LLM + ASR？哪些必须上云？ | 🔴 高 |
| 错过硬件申请后，能否补救？还是必须自购？ | 🟡 中 |
| 比赛评分如何体现"充分依托平台能力"？ | 🟡 中 |
| 诈骗电话录音数据从哪来？（TTS生成 vs B站爬取 vs 自己录制） | 🟡 中 |
| 训练人员是谁？模型包发给谁？ | 🟡 中 |

---

## 十、快速命令速查

```cmd
:: 激活 SenseVoice 环境
conda activate ssv

:: 清理代理
set HTTP_PROXY= && set HTTPS_PROXY= && set http_proxy= && set https_proxy=

:: 验证模型包存在
dir E:\sensevoice_model.tar.gz

:: 查看当前所有文件
dir E:\老人跌倒\ /B

:: 验证 Python 版本
python --version

:: 查看 conda 环境列表
conda env list
```
