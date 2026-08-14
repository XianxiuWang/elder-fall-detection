# innovation_task.md — 创新路线规划追踪文件 (v6.2+)

**所属**: 老人跌倒项目 — 创新路线规划专项子会话
**当前 E2E 版本**: v6.1 (多人检测 + 个性化步态基线)
**本文件**: 专项子会话的工作产出/决策/进度追踪 (永久保留)
**最后更新**: 2026-08-03

---

## 一、我的专属任务范围

为项目规划并推进 v6.2 及之后的创新方向，兼顾挑战杯评委视角：

| 方向 | 版本 | 创新类型 | 本质 |
|------|------|---------|------|
| 方向二 | v6.2 | 前兆动作识别 | "踉跄"状态机 → 跌倒前2-5秒语音询问 |
| 方向三 | v6.3 | 行为画像 | "今天和平时有什么不同" (时间序列统计, 无新模型) |
| 方向四 | v7.0 | 暖陪伴交互 | "你还好吗？"双向语音 + 个性化关怀 + 体验创新 |

我的职责：
1. 阅读 src/ 现有模块，避免重复造轮子
2. 输出创新方案设计（技术思路、数据需求、与现有系统衔接）
3. 逐步实现方向二/三的原型代码
4. 遵守项目踩坑记录（GBK终端禁用emoji、新CMD清代理、长任务不用conda run、PowerShell不支持&&）

---

## 二、现状盘点 (2026-08-03 已读源码)

### 已有模块 (src/)
| 文件 | 用途 | 与我的方向关系 |
|------|------|----------------|
| e2e_fall_monitor.py | 主入口 v6.1 | 集成点 (PersonState) |
| personalized_baseline.py | 个性化步态基线 | 方向二的特征来源 |
| gait_trend.py | GaitSample + 指标提取 | 数据源 |
| fall_early_warning.py | 6指标跌倒早期预警 | **方向二的核心原料** |
| fall_predictor.py | 跌倒风险预测 | 告警衔接 |
| ml_6class_detector.py | LightGBM 6类分类器 | 踉跄状态机的输入 |
| alert_manager.py | 告警通知 (winsound/pygame音频) | 语音询问的扩音通道 |
| multi_person.py | YOLOv8n多人检测 | 每人独立检测 |

### 关键发现 (重要! 避免重复造轮子)

1. **FallEarlyWarning 已经有 6 个"预跌倒"指标**：
   COM摇摆加速、步宽变异度、躯干倾斜角速度、支撑面缩小率、手臂扑腾度、步态节律紊乱。
   它已经输出 `pre_fall_risk` + `alert_level`。**方向二不应该从零做踉跄检测，而是在此基础上加"状态机"包装**。

2. **FallEarlyWarning 是"连续评分"，不是"离散状态"**：
   它每帧输出一个风险分，但没有"踉跄"这个**离散事件**概念。方向二的价值 = 把连续风险流 → 离散"踉跄前兆事件"，并响应式触发语音。

3. **AlertManager 只有"告警声" (beep)，没有自然语音**：
   `_play_alert()` 用的是 pygame合成beep / winsound beep。**方向二要加"你还好吗？" TTS 语音**，这是一个全新的通道（区别于告警）。

4. **audio 后端可用**: winsound 在 Windows 已验证可用。TTS 可走 3 条路：
   - 离线合成 WAV + 播放（零依赖，答辩演示最稳）
   - `pyttsx3`（若已装）
   - Edge TTS（在线，需代理注意，演示不推荐）

5. **集成点明确**：`e2e_fall_monitor.py` 的 `PersonState` 每人独立实例。方向二/三的模块应**挂在 PersonState 上**，每帧喂数据。

---

## 三、创新方案设计

### 方向二 (v6.2): 前兆动作识别 — "踉跄状态机" → "你还好吗？"

**创新点** (答辩卖点): 从"检测摔倒"(已发生, 拼技术) 升级为 "预测前兆"(未发生, 抢时间)。
把"摔倒"这个**瞬时事件**的检测，升级为"踉跄→失衡→(可能)摔倒"这个**过程**的识别 + 关怀式响应。

**核心技术**: `StumblePrecursorAnalyzer` (踉跄前兆状态机)

```
输入 (每帧, 挂PersonState):
├── FallEarlyWarning.pre_fall_risk (已有6指标融合分)
├── ML6ClassDetector.class (走路/站立/坐下...)
├── 新鲜度/连续帧数 (过滤冷启动)
└── PersonalBaseline的个人偏差 (可选加权, 加分项)

状态机 (4态):
  IDLE(正常) --[pre_fall≥35持续2帧]--> WATCH(关注)
  WATCH --[pre_fall≥55持续2帧]--> STUMBLING(踉跄预触发)
  STUMBLING --[多指标投票≥3 或 pre_fall≥70]--> DANGER(即将跌倒) ★触发语音
  DANGER --[ML判Fall 或 高度骤降]--> FALL_DETECTED(已跌倒, 移交告警)
  任意态 --[pre_fall<阈持续3秒]--> IDLE(恢复)

触发行为:
  进入 DANGER → 播放"你还好吗？"语音 (2-5秒前瞻窗口) + 屏幕关怀横幅
  进入 FALL_DETECTED → 移交 AlertManager 告警 (非语音, 是救援警报)
```

**关键区别**: "你还好吗？"是**暖关怀**（询问而非报警）→ v7.0 暖陪伴的前身。真正摔倒 → **救援警报**。两种响应是不同的。

**可解释性输出** (答辩要能讲): 每次DANGER触发输出 `StumbleEvent`:
```python
@dataclass
class StumbleEvent:
    person_id: str
    state: str             # WATCH/STUMBLING/DANGER
    pre_fall_risk: float
    active_indicators: List[str]   # 哪些指标在恶化
    duration_sec: float    # 踉跄持续多久
    frame_start: int / frame_trigger: int
    voice_asked: bool      # 是否已播报过
    ts: float
```

**数据需求**: 无新训练数据。需要的是带踉跄段的视频做**演示验证**。
（用 `人类摔倒五花八门.mp4` 验证 pre_fall 曲线是否能提前于摔倒抬升 → 这是答辩数据。）

### 方向三 (v6.3): 行为画像 — "今天和平时有什么不同"

**创新点**: 不训练新模型，把已有检测结果变成**长期行为时间线**，让系统能回答"今天和平时有什么不同"。

**核心技术**: `BehaviorProfile` (行为画像)
```
输入 (每帧挂PersonState):
├── ML6ClassDetector.class (走路/坐/站/躺)
├── 时间戳
└── (可选) gait指标

功能:
  1. 分桶统计: 每30分钟一个bucket, 记录各行为占比
  2. 行为节奏: 起床/入睡时间估算, 活动高峰
  3. "今天 vs 平时"异常检测: 用非参数统计(如 今天各行为分布 vs 过去7天滚动分布 的JS散度/比率)
  4. 每日摘要: 自然语言 "今天你走了X分钟, 比平时少了Y%"
  5. 持久化: JSON 到 behavior_profiles/
```

**答辩卖点**: 系统不只是"报警器"，而是**长期陪伴的行为分析师**。没有新模型 = 工程成本低、可解释、评委认可"用心程度"。

### 方向四 (v7.0): 暖陪伴交互

**创新点**: 人性化设计。老人跌倒预警系统最大的痛点是**"被监视感"**"—"监视"让人抗拒，但"陪伴"让人接受。从报警器 → 暖陪伴。

```
理念: "不是监视你, 是陪着你"
  1. 输入侧: 摄像头+语音(未来) → 理解老人行为与状态
  2. 决策侧: 行为画像 (方向三) + 踉跄检测 (方向二) + 基线 (v6.1)
  3. 输出侧: 
     a. 关怀语音 "你还好吗?" (踉跄时) ← 方向二
     b. 个性化日常问候 "今天你比平时少走了, 记得起来活动下" ← 方向三数据
     c. 家属通知 (真正危险时) ← 告警
  4. 话术层: 预设多套温柔话术池, 随机/按情境选择, 避免机械重复
```

**答辩价值**: 这是**体验创新**，评委能直观感受到项目"以人为本"。很多挑战杯作品技术强但冷冰冰，暖陪伴是差异化亮点。

---

## 四、实施计划 (里程碑)

- [x] M0: 读源码、盘点现状、写本追踪文件 (2026-08-03)
- [x] M1: 设计并实现 `stumble_precursor.py` (方向二核心) + 自测
- [x] M2: 把踉跄状态机集成到 e2e_fall_monitor.py (PersonState挂载)
- [x] M3: 实现 `behavior_profile.py` (方向三核心) + 自测
- [x] M4: 集成方向三 + 每日摘要输出
- [x] M5: 设计 v7.0 暖陪伴交互 (含话术池 + 与方向二三的衔接)
- [x] M6: 答辩PPT创新点包装文档 (评委视角)

> 注: v7.0 的"追问-应答完整循环"集成到主循环 (CareCompanion 深度接线) 列为后续待办,
> 原型代码 care_companion.py 已就绪且自测通过。

---

## 五、踩坑备忘 (专项会话专属)

- Windows GBK 终端: **代码里不用 emoji**, 输出用英文/中文标点
- 每次新 CMD: 先清代理 `set HTTP_PROXY=` 等四条
- 长任务: 不用 conda run; 先 `conda activate fall`
- PowerShell 脚本: 不支持 `&&`, 用 `;` 或分开写
- 我这里的 exec 是 PowerShell (不是 cmd): `dir` → `Get-ChildItem`, 用 `;` 分隔
- 模型/音频: 不依赖在线下载; TTS 演示用离线WAV/pyttsx3最稳

---

## 六、进度日志

### 2026-08-03
- 通读 HANDOFF_2026-08-02.md、ml_6class_detector.py、personalized_baseline.py、gait_trend.py、e2e_fall_monitor.py、fall_early_warning.py、alert_manager.py
- 确认 baselines/ 已有大量 Pxxx.json → v6.1 已在真实视频跑过 (多人场景)
- 设计完 4 个方向的技术方案 (见第三节)
- 确认方向二不要重复造轮子: 踉跄状态机应包装在 FallEarlyWarning 之上
- **M1 完成**: 新增 `src/stumble_precursor.py` (踉跄状态机 + 暖关怀语音)
  - `StumblePrecursorAnalyzer`: 4态状态机 IDLE→WATCH→STUMBLING→DANGER→FALL_DETECTED
  - `CareVoice`: Windows SAPI 中文TTS (离线, win32com已可用), 附带线程安全+话术池
  - `StumbleEvent`: 可解释事件 (答辩可打印)
  - 自测通过: 正常不误报 / 踉跄触发DANGER+语音 / 恢复IDLE
  - SAPI语音实测: 后端=sapi, 已成功发声 "你还好吗?"
- **M3 完成**: 新增 `src/behavior_profile.py` (方向三行为画像)
  - `BehaviorProfile`: 30分钟分桶 / 6类行为时间线 / "今天vs平时"JS散度 / 自然语言摘要
  - `DailyBehavior` / `BehaviorInsight` 数据类
  - 自测通过: 今天走路-88%被检出, JS散度0.15, 摘要生成, 持久化OK
- **M2+M4 完成**: 已将两个模块集成到 `e2e_fall_monitor.py`
  - PersonState.__slots__ + __init__ 挂载 care_voice / stumble / behavior
  - _process_person 每帧喂踉跄状态机 + 行为画像, 并设置 result 各字段
  - _draw_multi_panel 显示 Stumble:{state} 和 Behave:walk±%
  - 集成自测通过: fall 环境 (Py3.10) 下 PersonState 管线正确
  - 编译OK / import链OK
- **关键发现**: fall 环境原本无 win32com (SAPI不可用) → 跑 `pip install pywin32` 后可用 (注意: 我改了环境, 已记录)
  - CareVoice 现在支持 3 后端: 进程内SAPI / 子进程SAPI桥接 / winsound
- **M5 完成**: 新增 `src/care_companion.py` (v7.0 暖陪伴交互原型)
  - 3类关怀输出: 踉跄关怀(v6.2源) / 行为画像个性化关怀(v6.3源) / 陪伴应答(双向)
  - 双向交互: begin_greeting → 倾听窗口 → ok(老人活动=我没事)/escalate(升级关怀)
  - 无ASR也能用行为隐式反馈; 话术池轮换避免机械重复
  - 自测通过
- **M6 完成 (答辩包装)**: 新增 `交接文档/答辩创新点包装_2026-08-03.md`
  - 一句话主线: "报警器 → 暖陪伴"
  - 创新路线图讲故事逻辑 + 三个评委必问问题答案
  - 三大创新详解 + 答辩金句 + 技术可信度应对 + PPT结构
- **全部里程碑 M0-M6 完成**。本次专项会话交付: 3个创新模块 + 集成 + 答辩包装。

---

## 七、本次交付文件总览 (2026-08-03)

| 文件 | 方向 | 状态 |
|------|------|:--:|
| src/stumble_precursor.py | v6.2 踉跄状态机 + 暖语音 | ✅ 自测+集成过 |
| src/behavior_profile.py | v6.3 行为画像 | ✅ 自测+集成过 |
| src/care_companion.py | v7.0 暖陪伴交互 | ✅ 自测过 |
| src/e2e_fall_monitor.py | 主入口 (集成两模块) | ✅ 编译+导入过 |
| 交接文档/innovation_task.md | 本文件 (追踪) | ✅ |
| 交接文档/答辩创新点包装_2026-08-03.md | 答辩PPT大纲 | ✅ |

### 待办 (需用户在真机跑)
- [ ] 用踉跄视频跑 v6.2, 截"摔倒前已播报"关键帧
- [ ] 连续几天收集行为画像 → "今天vs平时"真实对比图
- [ ] 确认 SAPI 语音在真机听得见 (fall环境已装pywin32)
- [ ] v7.0 完整集成到主循环 (追问-应答循环) — 原型已就绪
