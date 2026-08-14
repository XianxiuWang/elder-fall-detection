#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stumble_precursor.py — 踉跄前兆识别 + 暖关怀语音 (v6.2)
=====================================================
方向二核心创新: 从"检测摔倒"(已发生) 升级为 "识别踉跄前兆"(未发生)。

设计思想:
  不重复造轮子。复用 fall_early_warning.py 的六指标连续风险分,
  在其之上增加一个"离散状态机"——把"连续风险流"翻译成
  "踉跄前兆事件", 并触发"你还好吗?"暖关怀语音。

     IDLE(正常) --[pre_fall>=35 持续2帧]--> WATCH(关注)
     WATCH --[pre_fall>=55 持续2帧]--> STUMBLING(踉跄预触发)
     STUMBLING --[指标投票>=3 或 pre_fall>=70]--> DANGER(即将跌倒)  ★触发语音
     DANGER --[ML判Fall 或 质心骤降]--> FALL_DETECTED(已跌倒, 移交告警)
     (任意态) --[pre_fall<恢复阈 持续3秒]--> IDLE(恢复)

关键区别 (答辩卖点):
  - "你还好吗?" = 暖关怀语音 (询问而非报警) → v7.0 暖陪伴前身
  - 真正摔倒 (FALL_DETECTED) = 救援警报 (移交 AlertManager, 非本模块)
  两种响应性质完全不同。

依赖:
  - fall_early_warning.FallEarlyWarning  (六指标连续风险)
  - ml_6class_detector.DetectionResult   (走路/站立等类别)

用法 (挂到 e2e_fall_monitor.PersonState):
    from .stumble_precursor import StumblePrecursorAnalyzer, CareVoice

    voice = CareVoice(language="zh-CN")          # SAPI TTS
    stumble = StumblePrecursorAnalyzer(
        early_warn=state.early_warning,          # 复用引擎
        voice=voice,
        person_id=f"P{person_id}")
    event = stumble.update(lm, ml_result, frame_idx, fps)
    if event and event.state == "DANGER":
        # 已在 DANGER 自动播"你还好吗?"
        pass
    elif event and event.state == "FALL_DETECTED":
        # 移交告警 (非本模块职责), 这里只是标记
        pass
"""

import os
import sys
import time
import threading
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field

# MediaPipe 关键点索引
_LEFT_HIP, _RIGHT_HIP = 23, 24
_NOSE = 0


# ════════════════════════════════════════════════════════
# 暖关怀语音 (CareVoice)
# ════════════════════════════════════════════════════════

class CareVoice:
    """
    暖关怀语音层 — 播放"你还好吗?"等关怀话语。
    与 AlertManager 的"告警beep"不同: 这是自然语音、是关怀而非报警。

    后端优先级:
      1. Windows SAPI (win32com) — 自然中文TTS, 离线, 演示最稳  [推荐]
      2. 本地WAV文件播放 (winsound) — 若已预生成 wav
      3. 静默 (无后端)

    线程安全: 语音播放放到独立线程, 不阻塞主检测循环。
    """

    # 关怀话术池 (答辩可展示"人性化")
    WARM_PHRASES_ZH = [
        "你还好吗？慢慢来，不着急。",
        "哎哟，小心点，站稳了。",
        "要不要坐下歇一会儿？",
        "没关系，我在这里。",
    ]
    WARM_PHRASES_EN = [
        "Are you okay? Take your time.",
        "Careful, watch your step.",
        "Do you want to sit down and rest?",
        "It's okay, I'm here with you.",
    ]

    # 已知带 SAPI (win32com) 的 Python 解释器候选 (子进程桥接用)
    KNOWN_SAPI_PYTHONS = [
        r"D:\Anaconda3\python.exe",      # base Anaconda 实测有 win32com
        r"D:\Anaconda3\envs\fall\python.exe",
    ]

    def __init__(self, language: str = "zh-CN",
                 phrase_pool: Optional[List[str]] = None,
                 use_thread: bool = True,
                 on_play: Optional[Callable[[str, str], None]] = None,
                 subprocess_python: Optional[str] = None):
        """
        Args:
            language: "zh-CN" 或 "en-US" (选择话术池)
            phrase_pool: 自定义话术池 (覆盖默认)
            use_thread: 是否在线程中播放 (避免阻塞主循环)
            on_play: 回调 (text, backend), 便于记录"播了什么"
            subprocess_python: 指定用于子进程TTS桥接的python; None自动探测
        """
        self.language = language
        self.use_thread = use_thread
        self.on_play = on_play
        self._lock = threading.Lock()
        self._last_phrase_time = 0.0
        self._subprocess_python = subprocess_python or self._find_sapi_python()

        # 话术池 (随机/循环选择, 避免机械重复)
        if phrase_pool is not None:
            self._phrases = list(phrase_pool)
        elif language.lower().startswith("zh"):
            self._phrases = list(self.WARM_PHRASES_ZH)
        else:
            self._phrases = list(self.WARM_PHRASES_EN)
        self._phrase_idx = 0

        # 后端探测
        self._backend = self._detect_backend()
        self._sapi = None
        self._sapi_voice_ready = False
        if self._backend == "subprocess":
            self._sapi_voice_ready = True  # 子进程桥接视为语音可用
        elif self._backend == "sapi":
            self._init_sapi()

    # ── 寻找带 SAPI 的 python (子进程桥接) ──
    @classmethod
    def _find_sapi_python(cls) -> Optional[str]:
        """找一个当前进程可用的、带 win32com 的 python"""
        candidates = [sys.executable] + cls.KNOWN_SAPI_PYTHONS
        for py in candidates:
            if not py or not os.path.exists(py):
                continue
            # 当前进程本身有 SAPI → 直接进程内
            if py == sys.executable:
                try:
                    import win32com.client  # noqa
                    return None  # 进程内已够用
                except ImportError:
                    continue
            # 用 --check 探测子进程是否有 win32com (快速)
            try:
                import subprocess
                r = subprocess.run(
                    [py, "-c", "import win32com.client"],
                    capture_output=True, timeout=8)
                if r.returncode == 0:
                    return py
            except Exception:
                continue
        return None

    # ── 后端探测 ──
    def _detect_backend(self) -> str:
        if sys.platform == "win32":
            try:
                import win32com.client  # noqa
                return "sapi"
            except ImportError:
                pass
        # 进程内无SAPI, 但有子进程桥接 → 也算可用语音
        if self._subprocess_python is not None:
            return "subprocess"
        try:
            import winsound  # noqa
            return "winsound"
        except ImportError:
            return "none"

    def _init_sapi(self):
        """初始化 SAPI。静默失败 (不崩溃)。"""
        try:
            import win32com.client
            self._sapi = win32com.client.Dispatch("SAPI.SpVoice")
            # 尝试选中文声线
            try:
                voices = self._sapi.GetVoices()
                zh_voice = None
                for i in range(voices.Count):
                    v = voices.Item(i)
                    desc = v.GetDescription()
                    if self.language.lower().startswith("zh") and ("Chinese" in desc or "中文" in desc or "HuiHui" in desc):
                        zh_voice = v
                        break
                    if self.language.lower().startswith("en") and "English" in desc:
                        zh_voice = v
                        break
                if zh_voice is not None:
                    self._sapi.Voice = zh_voice
            except Exception:
                pass
            self._sapi.Rate = -1   # 稍慢, 对老人更友好
            self._sapi.Volume = 100
            self._sapi_voice_ready = True
        except Exception as e:
            print(f"  [CareVoice] SAPI 初始化失败: {e}, 回退 winsound")
            self._backend = "winsound" if self._backend == "sapi" else self._backend

    @property
    def available(self) -> bool:
        return self._sapi_voice_ready or self._backend == "winsound"

    # ── 播报 ──
    def say_care(self, phrase: Optional[str] = None,
                 min_interval: float = 8.0) -> bool:
        """
        播放一条关怀话语。
        Args:
            phrase: 指定话术; None 则从话术池轮换
            min_interval: 最小播放间隔 (秒), 防止语音轰炸
        Returns:
            是否真的播放了
        """
        now = time.time()
        if now - self._last_phrase_time < min_interval:
            return False
        self._last_phrase_time = now

        text = phrase if phrase is not None else self._next_phrase()

        if self.on_play:
            try:
                self.on_play(text, self._backend)
            except Exception:
                pass

        if self.use_thread:
            t = threading.Thread(target=self._play, args=(text,), daemon=True)
            t.start()
        else:
            self._play(text)
        return True

    def _next_phrase(self) -> str:
        phrase = self._phrases[self._phrase_idx % len(self._phrases)]
        self._phrase_idx += 1
        return phrase

    def _play(self, text: str):
        """实际播放 (后台线程内调用)"""
        # 1. 进程内 SAPI
        if self._sapi_voice_ready and self._sapi is not None:
            try:
                with self._lock:
                    self._sapi.Speak(text, 1)  # 1 = 异步
                return
            except Exception:
                pass

        # 2. 子进程 SAPI 桥接 (当前进程无win32com, 用别的python说话)
        if self._backend == "subprocess" and self._subprocess_python is not None:
            try:
                import subprocess
                # 文本经临时文件传递, 避免 shell 转义问题
                tmp = os.path.join(os.environ.get("TEMP", "."),
                                   f"_carevoice_{os.getpid()}_{abs(hash(text)) % 10000}.txt")
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(text)
                code = (
                    "import win32com.client, sys;"
                    "t=open(sys.argv[1],encoding='utf-8').read();"
                    "v=win32com.client.Dispatch('SAPI.SpVoice');"
                    "v.Rate=-1; v.Speak(t)"
                )
                subprocess.run([self._subprocess_python, "-c", code, tmp],
                               capture_output=True, timeout=15)
                try:
                    os.remove(tmp)
                except Exception:
                    pass
                return
            except Exception:
                pass

        # 3. winsound 柔和短音 (无TTS时)
        if self._backend == "winsound":
            try:
                import winsound
                winsound.Beep(1200, 120)
            except Exception:
                pass

    def speak_file(self, wav_path: str, min_interval: float = 8.0) -> bool:
        """播放预生成 WAV (离线最稳, 用于答辩演示)"""
        now = time.time()
        if now - self._last_phrase_time < min_interval:
            return False
        self._last_phrase_time = now
        if not os.path.exists(wav_path):
            return False
        try:
            import winsound
            winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return True
        except Exception:
            return False

    def info(self) -> str:
        return (f"CareVoice[backend={self._backend}, "
                f"sapi_ready={self._sapi_voice_ready}, lang={self.language}]")


# ════════════════════════════════════════════════════════
# 踉跄前兆事件
# ════════════════════════════════════════════════════════

@dataclass
class StumbleEvent:
    """一次踉跄前兆事件 (可解释, 答辩可打印)"""
    person_id: str = ""
    state: str = "IDLE"            # IDLE/WATCH/STUMBLING/DANGER/FALL_DETECTED
    pre_fall_risk: float = 0.0     # 复用 FallEarlyWarning 的连续风险
    active_indicators: List[str] = field(default_factory=list)  # 恶化的核心指标
    stale_seconds: float = 0.0     # 踉跄状态已持续秒数 (DANGER触发时=前瞻窗口)
    frame_start: int = 0           # 进入当前状态的帧号
    frame_trigger: int = 0         # 触发DANGER的帧号
    voice_asked: bool = False      # 是否已播"你还好吗?"
    is_transition: bool = False    # 本帧是否发生了状态跳转
    direction_hint: str = ""       # "forward" 前扑 / "backward" 后仰 / "side" 侧倾
    ts: float = 0.0

    def summary(self) -> str:
        ind = ",".join(self.active_indicators) if self.active_indicators else "-"
        return (f"[{self.person_id}] {self.state} risk={self.pre_fall_risk:.0f} "
                f"ind=[{ind}] dur={self.stale_seconds:.1f}s "
                f"voice={self.voice_asked}")


# ════════════════════════════════════════════════════════
# 踉跄前兆状态机
# ════════════════════════════════════════════════════════

class StumblePrecursorAnalyzer:
    """
    踉跄前兆状态机 v1.0
    包装 FallEarlyWarning (连续风险) → 离散状态 + 语音触发。

    状态转移阈值 (可调):
      - 进入WATCH:  pre_fall_risk >= WATCH_RISK 持续 STAY_FRAMES 帧
      - 进入STUMBLING: pre_fall_risk >= STUMBLE_RISK 持续 STAY_FRAMES 帧
      - 进入DANGER:  指标投票>=3 或 pre_fall_risk>=DANGER_RISK
      - 恢复IDLE:   pre_fall_risk < RECOVER_RISK 持续 RECOVER_FRAMES 帧
    """

    # ── 阈值 ──
    WATCH_RISK = 35.0
    STUMBLE_RISK = 55.0
    DANGER_RISK = 70.0
    RECOVER_RISK = 25.0

    STAY_FRAMES = 2       # 进入中间态需要持续帧数 (防抖)
    RECOVER_FRAMES = 90   # 恢复需要帧数 (≈3秒@30fps, 防止轻微波动立即复位)
    VOTE_TRIGGER = 3      # DANGER: 多少个核心指标同时恶化

    # 核心恶化指标 (对应 FallEarlyWarning 六指标)
    CORE_INDICATORS = [
        ("COM摇摆加速",   "sway_accel_risk"),
        ("步宽变异",     "step_width_risk"),
        ("躯干倾斜速度", "trunk_tilt_vel_risk"),
        ("支撑面缩小",   "support_shrink_risk"),
        ("手臂扑腾",     "arm_flail_risk"),
        ("步态节律",     "gait_rhythm_risk"),
    ]

    def __init__(self, early_warn, voice: Optional[CareVoice] = None,
                 person_id: str = "P0", fps: float = 15.0):
        """
        Args:
            early_warn: FallEarlyWarning 实例 (复用六指标引擎)
            voice: CareVoice 实例 (播"你还好吗?"), 可为None(仅状态机)
            person_id: 人物ID
            fps: 帧率 (用于防抖帧数换算)
        """
        self.early_warn = early_warn
        self.voice = voice
        self.person_id = person_id
        self.fps = max(fps, 1.0)

        # 状态机状态
        self.state: str = "IDLE"
        self._state_since_frame: int = 0
        self._trigger_frame: int = 0

        # 防抖计数
        self._watch_streak = 0
        self._stumble_streak = 0
        self._recover_streak = 0

        # 统计 / 事件
        self.events: List[StumbleEvent] = []
        self.danger_count = 0
        self.fall_count = 0
        self._last_voice_time = 0.0

        # 历史 (供恢复判断 + 趋势)
        self._risk_history: List[float] = []

    # ════════════════════════════════════════════════════
    # 主接口
    # ════════════════════════════════════════════════════

    def update(self, landmarks, ml_result, frame_idx: int,
               elapsed: float) -> Optional[StumbleEvent]:
        """
        每帧调用。
        Args:
            landmarks: (33,4) MediaPipe 关键点
            ml_result: ML6ClassDetector.DetectionResult 或 None
            frame_idx: 全局帧号
            elapsed: 运行秒数
        Returns:
            状态机有"事件"时返回 StumbleEvent, 否则 None
            (仅当发生转移或DANGER时返回, 减少开销)
        """
        is_walking = True
        if ml_result is not None:
            # 非行走状态(坐下/躺)时, 降低踉跄误报
            if hasattr(ml_result, "class_id"):
                # 6类: Fall=0 SitDown=1 StandUp=2 Walking=3 WakeUp=4 Standing=5
                if ml_result.class_id in (1, 4):
                    is_walking = False

        # 复用 FallEarlyWarning 的连续风险
        warn_report = self.early_warn.update(
            landmarks, elapsed=elapsed, is_walking=is_walking)
        pre_risk = warn_report.pre_fall_risk
        self._risk_history.append(pre_risk)
        if len(self._risk_history) > 300:
            self._risk_history = self._risk_history[-200:]

        # 提取恶化指标
        active_indicators = [
            name for name, attr in self.CORE_INDICATORS
            if getattr(warn_report, attr, 0.0) >= 50.0
        ]

        # 跌倒方向提示 (基于质心与骨盆高度突降)
        direction = self._infer_direction(landmarks, ml_result)

        # ── 状态转移逻辑 ──
        event = None
        prev_state = self.state

        if self.state == "IDLE":
            if pre_risk >= self.WATCH_RISK:
                self._watch_streak += 1
            else:
                self._watch_streak = 0
            if self._watch_streak >= self.STAY_FRAMES:
                self._set_state("WATCH", frame_idx)
                event = self._make_event(pre_risk, active_indicators,
                                         frame_idx, direction, transition=True)

        elif self.state == "WATCH":
            if pre_risk >= self.STUMBLE_RISK:
                self._stumble_streak += 1
            else:
                self._stumble_streak = 0
            if self._stumble_streak >= self.STAY_FRAMES:
                self._set_state("STUMBLING", frame_idx)
                event = self._make_event(pre_risk, active_indicators,
                                         frame_idx, direction, transition=True)
            elif pre_risk < self.RECOVER_RISK:
                self._recover_streak += 1
                if self._recover_streak >= self.RECOVER_FRAMES:
                    self._set_state("IDLE", frame_idx)
                    event = self._make_event(pre_risk, active_indicators,
                                             frame_idx, direction, transition=True)
            else:
                self._recover_streak = 0

        elif self.state == "STUMBLING":
            # 达到 DANGER: 足够指标恶化 或 风险极高
            if (len(active_indicators) >= self.VOTE_TRIGGER
                    or pre_risk >= self.DANGER_RISK):
                self._set_state("DANGER", frame_idx)
                self.danger_count += 1
                self._trigger_frame = frame_idx
                event = self._make_event(pre_risk, active_indicators,
                                         frame_idx, direction, transition=True)
                # ★ 触发"你还好吗?"暖关怀语音
                if self.voice is not None and self.voice.available:
                    asked = self.voice.say_care()
                    event.voice_asked = asked
                # 已跌倒 (直接Fall类别)
                if ml_result is not None and getattr(ml_result, "is_fall", False):
                    self._set_state("FALL_DETECTED", frame_idx)
                    self.fall_count += 1
                    event.state = "FALL_DETECTED"
            elif pre_risk < self.RECOVER_RISK:
                self._recover_streak += 1
                if self._recover_streak >= self.RECOVER_FRAMES:
                    self._set_state("IDLE", frame_idx)
                    event = self._make_event(pre_risk, active_indicators,
                                             frame_idx, direction, transition=True)
            else:
                self._recover_streak = 0

        elif self.state == "DANGER":
            # DANGER 可能持续数帧; 只在前瞻窗口内停留, 之后要么已倒, 要么恢复
            if ml_result is not None and getattr(ml_result, "is_fall", False):
                self._set_state("FALL_DETECTED", frame_idx)
                self.fall_count += 1
                event = self._make_event(pre_risk, active_indicators,
                                         frame_idx, direction, transition=True)
                event.state = "FALL_DETECTED"
            elif pre_risk < self.RECOVER_RISK:
                self._recover_streak += 1
                if self._recover_streak >= self.RECOVER_FRAMES:
                    self._set_state("IDLE", frame_idx)
                    event = self._make_event(pre_risk, active_indicators,
                                             frame_idx, direction, transition=True)
            else:
                self._recover_streak = 0

        elif self.state == "FALL_DETECTED":
            # 已交接给告警; 一段时间后复位
            if pre_risk < self.RECOVER_RISK:
                self._recover_streak += 1
                if self._recover_streak >= self.RECOVER_FRAMES:
                    self._set_state("IDLE", frame_idx)
                    event = self._make_event(pre_risk, active_indicators,
                                             frame_idx, direction, transition=True)

        # 记录事件
        if event is not None:
            self.events.append(event)
            if len(self.events) > 200:
                self.events = self.events[-100:]

        return event

    # ════════════════════════════════════════════════════
    # 内部
    # ════════════════════════════════════════════════════

    def _set_state(self, new_state: str, frame_idx: int):
        self.state = new_state
        self._state_since_frame = frame_idx
        # 重置防抖计数
        self._watch_streak = 0
        self._stumble_streak = 0
        self._recover_streak = 0

    def _make_event(self, pre_risk, indicators, frame_idx,
                    direction, transition: bool) -> StumbleEvent:
        stale = (frame_idx - self._state_since_frame) / self.fps if self._state_since_frame else 0.0
        return StumbleEvent(
            person_id=self.person_id,
            state=self.state,
            pre_fall_risk=round(pre_risk, 1),
            active_indicators=indicators,
            stale_seconds=round(stale, 2),
            frame_start=self._state_since_frame,
            frame_trigger=frame_idx,
            is_transition=transition,
            direction_hint=direction,
            ts=time.time(),
        )

    def _infer_direction(self, lm, ml_result) -> str:
        """粗略判断跌倒方向 (前扑/后仰/侧倾)"""
        try:
            # 质心 (骨盆中点)
            hip_y = (lm[_LEFT_HIP, 1] + lm[_RIGHT_HIP, 1]) / 2
            nose_y = lm[_NOSE, 1]
            # 头部相对骨盆高度: 前扑时头部下探快
            head_ratio = (hip_y - nose_y) if hip_y > nose_y else 0.0
            if head_ratio < 0.05:
                return "forward"   # 头部几乎贴到骨盆 → 前扑
            return ""
        except Exception:
            return ""

    # ════════════════════════════════════════════════════
    # 查询
    # ════════════════════════════════════════════════════

    def get_status(self) -> dict:
        return {
            "state": self.state,
            "danger_count": self.danger_count,
            "fall_count": self.fall_count,
            "recent_risk": round(self._risk_history[-1], 1) if self._risk_history else 0.0,
            "voice": self.voice.info() if self.voice else "none",
        }

    def reset(self):
        self.state = "IDLE"
        self._state_since_frame = 0
        self._watch_streak = 0
        self._stumble_streak = 0
        self._recover_streak = 0
        self._risk_history.clear()


# ════════════════════════════════════════════════════════
# 自测 (纯逻辑, 不开摄像头, 不真放语音)
# ════════════════════════════════════════════════════════

def _test():
    print("=" * 60)
    print("踉跄前兆状态机 自测")
    print("=" * 60)

    from src.fall_early_warning import FallEarlyWarning
    from src.ml_6class_detector import DetectionResult
    import numpy as np

    # 用一个"哑"CareVoice (记录但不发声)
    class _SilentVoice(CareVoice):
        def __init__(self):
            self.calls = []
            super().__init__(use_thread=False)
        def say_care(self, phrase=None, min_interval=8.0):
            self.calls.append(phrase or "你还好吗?")
            return True

    voice = _SilentVoice()
    early_warn = FallEarlyWarning(fps=15.0)
    sm = StumblePrecursorAnalyzer(early_warn, voice=voice, person_id="T1", fps=15.0)

    # ── 阶段1: 正常步行 (应保持 IDLE) ──
    print("\n[阶段1] 正常步行 (IDLE)...")
    lm = np.random.rand(33, 4).astype(np.float32)
    lm[:, 1] = np.linspace(0.1, 0.7, 33)  # 站立高度
    for i in range(60):
        ml = DetectionResult(probs=np.zeros(6), class_id=3, class_name="Walking",
                             class_name_cn="走路", is_fall=False, is_standing=False,
                             fall_prob=0.01, standing_prob=0.2, fall_confidence=0.01,
                             fall_triggered=False, inference_done=True, inference_count=100)
        ev = sm.update(lm, ml, i, i / 15.0)
    state1 = sm.state
    print(f"  状态: {state1} (期望 IDLE)")
    assert state1 == "IDLE", f"期望 IDLE, 实际 {state1}"

    # ── 阶段2: 模拟踉跄 (逐步加大摇摆/倾斜) ──
    print("\n[阶段2] 模拟踉跄 (应触发 DANGER + 语音)...")
    np.random.seed(1)
    for i in range(120):
        # 逐渐恶化: 加大躯干倾斜, 增加摇摆, 手臂扑腾
        sway = min(0.25, i / 120 * 0.25) + np.random.normal(0, 0.02)
        lm = np.random.rand(33, 4).astype(np.float32)
        lm[:, 1] = np.linspace(0.1, 0.7, 33)
        lm[11, 0] = 0.5 + sway        # 左肩横移
        lm[12, 0] = 0.5 - sway        # 右肩横移
        lm[15, 0] = 0.3 - i / 120 * 0.3  # 左腕飞扑
        lm[16, 0] = 0.7 + i / 120 * 0.3  # 右腕飞扑
        lm[11:13, 1] = 0.3 - i / 120 * 0.15  # 肩部下探 → 前扑
        ml = DetectionResult(probs=np.zeros(6), class_id=3, class_name="Walking",
                             class_name_cn="走路", is_fall=False, is_standing=False,
                             fall_prob=0.05, standing_prob=0.1, fall_confidence=0.05,
                             fall_triggered=False, inference_done=True, inference_count=100)
        ev = sm.update(lm, ml, 60 + i, (60 + i) / 15.0)
        if ev:
            print(f"  [帧{60+i}] {ev.summary()}")

    print(f"\n  最终状态: {sm.state}")
    print(f"  DANGER 次数: {sm.danger_count}")
    print(f"  语音播报次数: {len(voice.calls)}")
    if voice.calls:
        print(f"  语音内容示例: {voice.calls[0]}")
    assert sm.danger_count >= 1, f"期望触发至少1次DANGER, 实际 {sm.danger_count}"
    assert len(voice.calls) >= 1, "期望至少播1次语音"

    # ── 阶段3: 恢复正常 (应回到 IDLE) ──
    print("\n[阶段3] 恢复正常 (应回到 IDLE)...")
    lm2 = np.random.rand(33, 4).astype(np.float32)
    lm2[:, 1] = np.linspace(0.1, 0.7, 33)
    for i in range(200):
        ml = DetectionResult(probs=np.zeros(6), class_id=3, class_name="Walking",
                             class_name_cn="走路", is_fall=False, is_standing=False,
                             fall_prob=0.01, standing_prob=0.3, fall_confidence=0.01,
                             fall_triggered=False, inference_done=True, inference_count=120)
        ev = sm.update(lm2, ml, 180 + i, (180 + i) / 15.0)
        if ev:
            print(f"  [帧{180+i}] {ev.summary()}")
    print(f"  最终状态: {sm.state} (期望 IDLE)")
    assert sm.state == "IDLE", f"期望恢复 IDLE, 实际 {sm.state}"

    print("\n" + "=" * 60)
    print("踉跄前兆状态机 自测全部通过!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    _test()
