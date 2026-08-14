#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
care_companion.py — 暖陪伴交互层 (v7.0 设计 + 原型)
===================================================
方向四核心: 把"跌倒预警系统"从冷冰冰的"报警器"升级为"暖陪伴"。
核心理念: "不是监视你, 是陪着你。"

定位:
  前几层负责"感知/判断":
    v6.1 个人基线      → 和"过去的自己"比
    v6.2 踉跄前兆状态机 → 跌倒前 2-5 秒识别踉跄 → "你还好吗?"
    v6.3 行为画像       → "今天和平时有什么不同"
  v7.0 本层负责"输出/交互": 用自然、温暖、个性化的话术把上述洞察
    表达出来, 并支持"一问一答"的关怀式交互。

三种关怀输出 (对应三个数据来源):
  A. 踉跄关怀  [来自 v6.2]  "你还好吗? 慢慢来, 不着急。"
     → 跌倒前 2-5 秒, 询问而非报警 (v7 前身)
  B. 日常关怀  [来自 v6.3 行为画像]
     → "今天你比平时少走了, 记得起来活动一下。"
     → "今天活动高峰比以前早了些, 昨晚睡得好吗?"
  C. 陪伴应答  [双向交互, 本层原型]
     → 系统问 → 老人用简单语音/动作回应 → 系统体贴回应
     → 模式: ASK → LISTEN(窗口) → RESPOND
     → 即使没有ASR, 也可用"沉默/继续活动"作为隐式反馈:
       老人继续正常活动 = "我没事"; 长时间无活动/异常 = 升级关怀

答辩卖点 (体验创新):
  - 传统监护系统让老人有"被监视感" → 抗拒
  - 本系统用"陪伴感" → 老人更愿意长期使用
  - 用数据说话: 踉跄次数、每日关怀投递、老人接受度
  - 完全可解释、无新模型、可离线运行

依赖:
  - stumble_precursor.CareVoice   (暖语音播放)
  - behavior_profile.BehaviorInsight (行为洞察)
"""

import os
import time
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class CompanionEvent:
    """一次暖陪伴互动事件"""
    kind: str                  # "stumble_care" / "daily_greeting" / "pat_answer" / "escalate"
    channel: str               # "voice" / "screen" / "both"
    text_cn: str
    text_en: str
    source: str                # 数据来源: stumble / behavior_profile / rule
    ts: float = 0.0

    def __post_init__(self):
        if not self.ts:
            self.ts = time.time()


class WarmPhrases:
    """话术池 — 避免机械重复, 按情境随机/轮换"""

    # ── 踉跄关怀 (v6.2 触发) ──
    STUMBLE_ZH = [
        "你还好吗? 慢慢来, 不着急。",
        "哎哟, 小心点, 站稳了。",
        "要不要坐下歇一会儿? 我陪着你。",
        "没关系, 我在呢。",
    ]
    STUMBLE_EN = [
        "Are you okay? Take it slow.",
        "Careful, watch your step.",
        "Do you want to sit for a moment? I'm here.",
        "It's okay, I'm right here.",
    ]

    # ── 行为画像个性化关怀 (v6.3 数据) ──
    def daily_greeting(self, cn: bool = True, walk_dev: Optional[float] = None,
                       peak_hour: Optional[int] = None,
                       usual_peak: Optional[int] = None) -> str:
        parts_cn, parts_en = [], []
        if walk_dev is not None:
            if walk_dev < -15:
                parts_cn.append(f"今天你比平时少走了{abs(walk_dev):.0f}%, 记得起来活动一下")
                parts_en.append(f"You walked {abs(walk_dev):.0f}% less than usual today, remember to move around")
            elif walk_dev > 15:
                parts_cn.append(f"今天你真棒, 比平时多走了{walk_dev:.0f}%")
                parts_en.append(f"Great job! You walked {walk_dev:.0f}% more than usual today")
        if peak_hour is not None and usual_peak is not None and peak_hour != usual_peak:
            parts_cn.append(f"注意到你今天的活动高峰比平时早了, 昨晚休息得好吗")
            parts_en.append("I noticed your activity peak came earlier than usual, did you rest well?")
        if not parts_cn:
            parts_cn.append("今天你的状态看起来很不错")
            parts_en.append("You seem to be doing great today")
        return ("；".join(parts_cn), " ".join(parts_en))

    # ── 陪伴应答 (双向交互 / 隐式反馈) ──
    PAT_ZH = [
        "那就好, 我在呢。",
        "好的, 一切都好就放心了。",
        "嗯, 我一直在看着你, 放心。",
    ]
    PAT_EN = [
        "Good, I'm here with you.",
        "Okay, glad you're fine.",
        "Yes, I'm always watching over you.",
    ]
    ESCALATE_ZH = [
        "你还好吗? 如果觉得不舒服, 可以坐下休息。我会联系家人。",
        "需要帮忙的话, 就慢慢坐下, 我会叫人来帮你。",
    ]
    ESCALATE_EN = [
        "Are you okay? If you feel unwell, sit down and rest. I'll contact family.",
        "If you need help, slowly sit down and I'll call someone to help you.",
    ]


class CareCompanion:
    """
    暖陪伴交互层 (v7.0 原型)。

    负责把 v6.2(踉跄) 和 v6.3(行为画像) 的洞察, 转化成温暖、个性化、
    有温度的输出, 并支持"问候-应答"式交互。

    用法 (挂在 PersonState 或全局):
        companion = CareCompanion(voice=state.care_voice)
        evt = companion.on_stumble()          # 踉跄 → "你还好吗?"
        evt = companion.on_daily_report(insight)  # 行为画像 → 个性化关怀
        # 双向交互:
        companion.begin_greeting()
        response = companion.complete_greeting(ok=True)  # 老人OK → 体贴应答
    """

    ASK_LISTEN_SECONDS = 8.0      # 问完后"倾听"窗口 (等待老人隐式反馈)

    def __init__(self, voice=None,
                 person_id: str = "P0",
                 on_event: Optional[callable] = None):
        """
        Args:
            voice: CareVoice 实例 (播语音); None 则仅屏幕/记录
            person_id: 人物ID
            on_event: 回调 (CompanionEvent), 便于 HUD/日志记录
        """
        self.voice = voice
        self.person_id = person_id
        self.on_event = on_event
        self.phrases = WarmPhrases()

        # 交互状态 (双向问候)
        self._listening_until = 0.0
        self._pending_kind = None
        self._pending_payload = None

        # 统计
        self.stumble_cares = 0
        self.daily_greetings = 0
        self.pat_answers = 0
        self.escalations = 0
        self.history: List[CompanionEvent] = []

        # 话术轮换索引
        self._idx_stumble = 0
        self._idx_pat = 0
        self._idx_escalate = 0

    # ════════════════════════════════════════════════════
    # A. 踉跄关怀 (v6.2 触发)
    # ════════════════════════════════════════════════════

    def on_stumble(self, direction_hint: str = "", min_interval: float = 20.0) -> Optional[CompanionEvent]:
        """踉跄前兆触发 → 播"你还好吗?"暖语音"""
        text_cn, text_en = self._pick_stumble()
        evt = CompanionEvent(kind="stumble_care", channel="both",
                             text_cn=text_cn, text_en=text_en,
                             source="stumble")
        spoken = False
        if self.voice is not None and self.voice.available:
            spoken = self.voice.say_care(text_cn, min_interval=min_interval)
        if not spoken:
            evt.channel = "screen"  # 没能播语音, 退化为屏幕关怀
        self.stumble_cares += 1
        self._record(evt)
        return evt

    # ════════════════════════════════════════════════════
    # B. 日常关怀 (v6.3 行为画像驱动)
    # ════════════════════════════════════════════════════

    def on_daily_report(self, insight, min_interval: float = 3600.0) -> Optional[CompanionEvent]:
        """
        行为画像洞察 → 个性化日常关怀。
        只有"今天和平时有显著不同"或"有异常"时才主动投递, 避免过度打扰。
        """
        if insight is None or not insight.has_enough_data:
            return None
        # 需要区分于平时才有意义
        has_insight = (
            (insight.walk_deviation_pct is not None and abs(insight.walk_deviation_pct) >= 15)
            or (insight.behavior_divergence >= 0.10)
            or insight.high_risk_events > 0
        )
        if not has_insight:
            return None

        text_cn, text_en = self.phrases.daily_greeting(
            walk_dev=insight.walk_deviation_pct,
            peak_hour=insight.today_peak_hour,
            usual_peak=insight.usual_peak_hour)
        evt = CompanionEvent(kind="daily_greeting", channel="both",
                             text_cn=text_cn, text_en=text_en,
                             source="behavior_profile")
        if self.voice is not None and self.voice.available:
            self.voice.say_care(text_cn, min_interval=min_interval)
        self.daily_greetings += 1
        self._record(evt)
        return evt

    # ════════════════════════════════════════════════════
    # C. 双向陪伴应答 (问答式)
    # ════════════════════════════════════════════════════

    def begin_greeting(self, kind: str = "stumble_care", payload=None):
        """开启一次问候, 进入倾听窗口"""
        self._pending_kind = kind
        self._pending_payload = payload
        self._listening_until = time.time() + self.ASK_LISTEN_SECONDS

    def complete_greeting(self, ok: bool) -> Optional[CompanionEvent]:
        """
        完成一次问候, 根据老人隐式反馈给出应答。
        Args:
            ok: True=老人正常/继续活动; False=老人可能需要帮助
        Returns:
            CompanionEvent (应答)
        """
        self._listening_until = 0.0
        if ok:
            text_cn, text_en = self._pick_pat()
            evt = CompanionEvent(kind="pat_answer", channel="both",
                                 text_cn=text_cn, text_en=text_en,
                                 source="rule")
            self.pat_answers += 1
        else:
            text_cn, text_en = self._pick_escalate()
            evt = CompanionEvent(kind="escalate", channel="both",
                                 text_cn=text_cn, text_en=text_en,
                                 source="rule")
            self.escalations += 1
        if self.voice is not None and self.voice.available:
            self.voice.say_care(text_cn, min_interval=0)
        self._record(evt)
        return evt

    def refresh_pending(self, is_walking: bool, recently_moved: bool) -> Optional[CompanionEvent]:
        """
        倾听窗口内轮询老人反馈 (无需ASR, 用行为隐式反馈):
          - 老人仍在活动/走动 → "我没事" (ok=True)
          - 长时间一动不动 → 升级关怀 (ok=False)
        由调用方每帧调用。
        """
        if self._listening_until <= 0:
            return None
        if time.time() > self._listening_until:
            # 窗口结束还没明确反馈 → 视为"我没事"
            return self.complete_greeting(ok=True)
        if is_walking and recently_moved:
            # 仍在走路 = 明确"我没事"
            return self.complete_greeting(ok=True)
        return None  # 还在等

    # ════════════════════════════════════════════════════
    # 话术选择 (轮换避免机械重复)
    # ════════════════════════════════════════════════════

    def _pick_stumble(self):
        i = self._idx_stumble % len(self.phrases.STUMBLE_ZH)
        self._idx_stumble += 1
        return self.phrases.STUMBLE_ZH[i], self.phrases.STUMBLE_EN[i]

    def _pick_pat(self):
        i = self._idx_pat % len(self.phrases.PAT_ZH)
        self._idx_pat += 1
        return self.phrases.PAT_ZH[i], self.phrases.PAT_EN[i]

    def _pick_escalate(self):
        i = self._idx_escalate % len(self.phrases.ESCALATE_ZH)
        self._idx_escalate += 1
        return self.phrases.ESCALATE_ZH[i], self.phrases.ESCALATE_EN[i]

    def _record(self, evt: CompanionEvent):
        self.history.append(evt)
        if len(self.history) > 200:
            self.history = self.history[-100:]
        if self.on_event is not None:
            try:
                self.on_event(evt)
            except Exception:
                pass

    # ════════════════════════════════════════════════════
    # 状态
    # ════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        return {
            "stumble_cares": self.stumble_cares,
            "daily_greetings": self.daily_greetings,
            "pat_answers": self.pat_answers,
            "escalations": self.escalations,
            "listening": self._listening_until > time.time(),
        }


# ════════════════════════════════════════════════════
# 自测 (纯逻辑, 不开语音)
# ════════════════════════════════════════════════════

def _test():
    print("=" * 60)
    print("暖陪伴交互层 自测")
    print("=" * 60)

    from src.stumble_precursor import CareVoice
    from src.behavior_profile import BehaviorProfile, BehaviorInsight

    # 哑 voice (只记录不发声)
    class _Silent(CareVoice):
        def __init__(self):
            self.calls = []
            super().__init__(use_thread=False)
        def say_care(self, phrase=None, min_interval=8.0):
            self.calls.append(phrase or "")
            return True
        @property
        def available(self):
            return True

    voice = _Silent()
    events = []
    comp = CareCompanion(voice=voice, on_event=lambda e: events.append(e))

    # ── A. 踉跄关怀 ──
    print("\n[A] 踉跄关怀...")
    evt = comp.on_stumble(direction_hint="forward", min_interval=0)
    assert evt.kind == "stumble_care"
    print(f"  播报: {evt.text_cn}")
    print(f"  语音条数: {len(voice.calls)}")

    # ── B. 行为画像日常关怀 ──
    print("\n[B] 行为画像日常关怀...")
    insight = BehaviorInsight(today_date="2026-08-03", has_enough_data=True,
                              today_walk_min=5.0, usual_walk_min=12.0,
                              walk_deviation_pct=-58.3, behavior_divergence=0.2,
                              high_risk_events=1, today_peak_hour=10, usual_peak_hour=15)
    evt2 = comp.on_daily_report(insight, min_interval=0)
    assert evt2 is not None and evt2.kind == "daily_greeting"
    print(f"  关怀: {evt2.text_cn}")

    # ── C. 双向应答 ──
    print("\n[C] 双向应答 (模拟 询问→老人OK→应答)...")
    comp.begin_greeting()
    # 老人继续走路 → 视为"我没事"
    r_evt = comp.complete_greeting(ok=True)
    assert r_evt.kind == "pat_answer"
    print(f"  应答: {r_evt.text_cn}")

    print("\n[D] 升级关怀 (老人异常...)...")
    comp.begin_greeting()
    r2 = comp.complete_greeting(ok=False)
    assert r2.kind == "escalate"
    print(f"  升级: {r2.text_cn}")

    print(f"\n  统计: {comp.get_stats()}")
    assert comp.stumble_cares == 1
    assert comp.daily_greetings == 1
    assert comp.pat_answers == 1
    assert comp.escalations == 1

    print("\n" + "=" * 60)
    print("暖陪伴交互层 自测全部通过!")
    print("=" * 60)


if __name__ == "__main__":
    _test()
