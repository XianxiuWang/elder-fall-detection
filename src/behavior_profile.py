#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
behavior_profile.py — 行为画像 (v6.3)
=====================================
方向三核心创新: "今天和平时有什么不同" (认知创新)。
不需要新模型, 把已有检测结果 (ML6Class 的类别流) 变成
长期行为时间线 + "今天 vs 平时" 异常检测。

概念:
  1. 行为分桶  : 每30分钟一个bucket, 统计各行为占比
  2. 行为节奏  : 估算起床/入睡时间, 活动高峰时段
  3. 今日异常  : 今天各行为分布 vs 过去滚动7天分布 的比较
                 (非参数统计: 比率 + 活动总量变化, 不假设分布)
  4. 每日摘要  : 自然语言 "今天你走了X分钟, 比平时少了Y%"
  5. 持久化    : JSON 到 behavior_profiles/

答辩卖点:
  - 系统不只是"报警器", 而是长期陪伴的"行为分析师"
  - 零新模型, 工程成本低, 结果可解释 (评委认可"用心")
  - 为 v7.0 暖陪伴提供"个性化日常问候"的数据底座

依赖:
  - ml_6class_detector.DetectionResult 的 class_id 流
  - 标准库 + (可选) numpy
"""

import os
import json
import time
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ── 六个行为类别 (与 ML6ClassDetector 一致) ──
CLASS_NAMES_CN = {
    0: "摔倒", 1: "坐下", 2: "站起", 3: "走路", 4: "睡醒", 5: "站立",
}
# 仅参与"活动画像"的核心行为 (排除了瞬时动作: 摔倒/站起)
_ACTIVITY_CLASSES = [1, 3, 5]          # 坐下/走路/站立
_SEDENTARY_CLASSES = [1, 4]            # 坐下/睡醒 = 久坐/躺卧
_MOVEMENT_CLASSES = [3]                # 走路


class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "item"):
            return obj.item()
        return super().default(obj)


@dataclass
class DailyBehavior:
    """一天的行为统计"""
    date: str = ""                       # YYYY-MM-DD
    bucket_seconds: int = 1800           # 30分钟
    buckets: Dict[int, Dict[str, float]] = field(default_factory=dict)  # bucket_idx → 各类别帧占比
    frames_per_class: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    total_frames: int = 0
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0

    def add_frame(self, class_id: int, timestamp: float):
        self.frames_per_class[class_id] += 1
        self.total_frames += 1
        if self.first_seen_ts == 0:
            self.first_seen_ts = timestamp
        self.last_seen_ts = max(self.last_seen_ts, timestamp)

        # 分桶累计 (只统计到bucket级别, 降噪)
        if class_id in _ACTIVITY_CLASSES:
            local = datetime.fromtimestamp(timestamp)
            day_start = datetime(local.year, local.month, local.day).timestamp()
            bucket_idx = int((timestamp - day_start) // self.bucket_seconds)
            b = self.buckets.setdefault(bucket_idx, defaultdict(float))
            b[class_id] += 1.0

    # ── 便捷统计 ──
    def class_ratio(self, class_id: int) -> float:
        return (self.frames_per_class.get(class_id, 0) / self.total_frames
                if self.total_frames else 0.0)

    def walking_minutes(self, fps: float = 30.0) -> float:
        """估算走路时长 (分钟)"""
        return self.frames_per_class.get(3, 0) / max(fps, 1.0) / 60.0

    def sedentary_minutes(self, fps: float = 30.0) -> float:
        """估算久坐/躺卧时长 (分钟)"""
        return sum(self.frames_per_class.get(c, 0) for c in _SEDENTARY_CLASSES) / max(fps, 1.0) / 60.0

    def activity_buckets(self) -> List[int]:
        """有活动的 bucket 索引 (当天从0开始)"""
        return sorted(self.buckets.keys())

    def peak_activity_hour(self) -> int:
        """活动最集中的小时 (当天)"""
        hour_frames = defaultdict(int)
        for bidx, counts in self.buckets.items():
            hour = (bidx * self.bucket_seconds) // 3600
            hour_frames[hour] += sum(counts.values())
        if not hour_frames:
            return -1
        return max(hour_frames, key=hour_frames.get)

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "bucket_seconds": self.bucket_seconds,
            "buckets": {str(k): dict(v) for k, v in self.buckets.items()},
            "frames_per_class": dict(self.frames_per_class),
            "total_frames": self.total_frames,
            "first_seen_ts": self.first_seen_ts,
            "last_seen_ts": self.last_seen_ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DailyBehavior":
        db = cls()
        db.date = d["date"]
        db.bucket_seconds = d.get("bucket_seconds", 1800)
        db.buckets = {int(k): defaultdict(float, v) for k, v in d.get("buckets", {}).items()}
        db.frames_per_class = defaultdict(int, {int(k): v for k, v in d.get("frames_per_class", {}).items()})
        db.total_frames = d.get("total_frames", 0)
        db.first_seen_ts = d.get("first_seen_ts", 0)
        db.last_seen_ts = d.get("last_seen_ts", 0)
        return db


@dataclass
class BehaviorInsight:
    """"今天 vs 平时"洞察报告"""
    today_date: str = ""
    has_enough_data: bool = False       # 今天数据是否够(>100帧)且平时有参照
    # 活动强度
    today_walk_min: float = 0.0
    usual_walk_min: float = 0.0         # 平时(7天滚动)平均走路分钟
    walk_deviation_pct: float = 0.0     # 走路时长偏差%
    # 行为分布差异 (JS散度, 0=完全一样)
    behavior_divergence: float = 0.0
    # 节奏变化
    today_peak_hour: int = -1
    usual_peak_hour: int = -1
    # 风险信号 (由外部基线提供, 画像聚合)
    high_risk_events: int = 0           # 今天DANGER/ALERT次数
    # 自然语言摘要
    summary_cn: str = ""
    summary_en: str = ""


class BehaviorProfile:
    """
    个人行为画像。
    记录每日行为时间线, 支持 "今天和平时有什么不同" 的洞察。

    用法:
        profile = BehaviorProfile(person_id="P1")
        profile.update(ml_result.class_id, timestamp)   # 每帧
        insight = profile.report_today()                 # 获取今日洞察
        profile.save()                                    # 持久化
    """

    HISTORY_DAYS = 7            # "平时"滚动窗口天数
    DEFAULT_BUCKET_SEC = 1800   # 30分钟分桶
    MIN_FRAME_TODAY = 100       # 今天最少帧数才出洞察
    MIN_USUAL_FRAMES = 200      # 平时参照最少帧数

    def __init__(self, person_id: str, save_dir: str = "behavior_profiles",
                 fps: float = 30.0, bucket_seconds: int = DEFAULT_BUCKET_SEC):
        self.person_id = person_id
        self.save_dir = save_dir
        self.fps = max(fps, 1.0)
        self.bucket_seconds = bucket_seconds
        self.days: Dict[str, DailyBehavior] = {}   # date → DailyBehavior
        self._today: Optional[DailyBehavior] = None
        self._today_date = ""
        self.high_risk_count_today = 0
        self.risk_reset_date = ""

    # ════════════════════════════════════════════════════
    # 主接口
    # ════════════════════════════════════════════════════

    def update(self, class_id: int, timestamp: Optional[float] = None) -> DailyBehavior:
        """
        每帧调用。
        Args:
            class_id: ML6Class 类别 (0-5)
            timestamp: 帧时间戳 (默认now)
        Returns:
            当日 DailyBehavior
        """
        ts = timestamp if timestamp is not None else time.time()
        today = self._get_today(ts)
        today.add_frame(class_id, ts)
        return today

    def register_risk_event(self, severity: str, timestamp: Optional[float] = None):
        """
        记录风险事件 (踉跄DANGER / 基线ALERT...), 计入今日画像。
        Args:
            severity: "DANGER"/"ALERT"/"WARN" 等
        """
        ts = timestamp if timestamp is not None else time.time()
        today = self._get_today(ts)
        # 只统计严重事件
        if severity in ("DANGER", "ALERT"):
            today = self._get_today(ts)
            self._ensure_risk_date(ts)
            self.high_risk_count_today += 1

    # ════════════════════════════════════════════════════
    # 洞察
    # ════════════════════════════════════════════════════

    def report_today(self, now: Optional[float] = None) -> BehaviorInsight:
        """
        生成 "今天和平时有什么不同" 洞察报告。
        """
        ts = now if now is not None else time.time()
        today = self._get_today(ts)
        today_date = self._date_str(ts)
        insight = BehaviorInsight(today_date=today_date)

        # 数据充分性
        if today.total_frames < self.MIN_FRAME_TODAY:
            insight.summary_cn = f"今天数据还在积累中 ({today.total_frames}/{self.MIN_FRAME_TODAY}帧)"
            return insight

        # ── 平时参照 (过去 N 天滚动平均) ──
        usual_frames = 0
        usual_walk_min = 0.0
        usual_peak_hours: Dict[int, int] = defaultdict(int)
        history_days = defaultdict(int)  # class -> 帧
        n_days = 0
        today_dt = datetime.fromtimestamp(ts).date()

        for date_str, db in self.days.items():
            if date_str == today_date:
                continue
            # 只取最近 HISTORY_DAYS
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                continue
            if (today_dt - d).days > self.HISTORY_DAYS:
                continue
            n_days += 1
            for c, cnt in db.frames_per_class.items():
                history_days[c] += cnt
            usual_frames += db.total_frames
            usual_walk_min += db.walking_minutes(self.fps)
            ph = db.peak_activity_hour()
            if ph >= 0:
                usual_peak_hours[ph] += 1

        if n_days == 0 or usual_frames < self.MIN_USUAL_FRAMES:
            insight.summary_cn = (f"还需要积累几天日常数据才能对比"
                                  f" ({n_days}天, {usual_frames}帧)")
            return insight

        insight.has_enough_data = True
        insight.usual_walk_min = round(usual_walk_min / n_days, 1)

        # ── 今天 vs 平时: 走路时长 ---
        today_walk = today.walking_minutes(self.fps)
        insight.today_walk_min = round(today_walk, 1)
        if insight.usual_walk_min > 0.5:
            insight.walk_deviation_pct = round(
                (today_walk - insight.usual_walk_min) / insight.usual_walk_min * 100.0, 1)
        else:
            insight.walk_deviation_pct = None

        # ── 行为分布差异 (JS散度) ──
        insight.behavior_divergence = self._js_divergence(today, history_days, n_days)

        # ── 活动高峰时段变化 ──
        insight.today_peak_hour = today.peak_activity_hour()
        if usual_peak_hours:
            insight.usual_peak_hour = max(usual_peak_hours, key=usual_peak_hours.get)

        # ── 风险事件合并 ──
        self._ensure_risk_date(ts)
        insight.high_risk_events = self.high_risk_count_today

        # ── 生成自然语言摘要 ──
        insight.summary_cn, insight.summary_en = self._build_summary(insight)
        return insight

    # ════════════════════════════════════════════════════
    # 内部
    # ════════════════════════════════════════════════════

    def _get_today(self, ts: float) -> DailyBehavior:
        today_date = self._date_str(ts)
        if self._today_date != today_date:
            # 跨天: 归档今天, 开启新一天
            if self._today is not None:
                self.days[self._today_date] = self._today
                # 裁剪历史
                self._prune_history()
            self._today_date = today_date
            self._today = DailyBehavior(
                date=today_date, bucket_seconds=self.bucket_seconds)
            self._ensure_risk_date(ts)
            self.high_risk_count_today = 0
        return self._today

    def _ensure_risk_date(self, ts: float):
        d = self._date_str(ts)
        if self.risk_reset_date != d:
            self.risk_reset_date = d
            self.high_risk_count_today = 0

    def _prune_history(self):
        """只保留最近 HISTORY_DAYS + 1 天, 控制内存"""
        cutoff = datetime.now().date() - timedelta(days=self.HISTORY_DAYS + 1)
        to_del = []
        for date_str in list(self.days):
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date()
                if d < cutoff:
                    to_del.append(date_str)
            except ValueError:
                to_del.append(date_str)
        for k in to_del:
            del self.days[k]

    @staticmethod
    def _date_str(ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")

    def _js_divergence(self, today: DailyBehavior,
                       history_days: Dict[int, int], n_days: int) -> float:
        """
        JS散度: 今天活动分布 vs 平时活动分布。
        0 = 完全一样; 越大 = 今天越反常。
        只用核心活动类别 (坐下/走路/站立)。
        """
        # 构建今天分布 (标准化)
        today_vec = []
        usual_vec = []
        total_today = 0
        total_usual = 0
        for c in _ACTIVITY_CLASSES:
            tc = today.frames_per_class.get(c, 0)
            uc = history_days.get(c, 0) / max(n_days, 1)
            today_vec.append(tc)
            usual_vec.append(uc)
            total_today += tc
            total_usual += uc

        if total_today == 0 or total_usual == 0:
            return 0.0

        p = [x / total_today for x in today_vec]
        q = [x / total_usual for x in usual_vec]

        # KL 散度 (加平滑)
        eps = 1e-9
        m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
        kl_pm = sum(pi * math.log((pi + eps) / (mi + eps)) for pi, mi in zip(p, m))
        kl_qm = sum(qi * math.log((qi + eps) / (mi + eps)) for qi, mi in zip(q, m))
        js = 0.5 * kl_pm + 0.5 * kl_qm
        return round(js, 4)

    def _build_summary(self, ins: BehaviorInsight) -> Tuple[str, str]:
        """生成自然语言摘要 (中/英)"""
        parts_cn = []
        parts_en = []

        # 走路量变化
        if ins.walk_deviation_pct is not None:
            if abs(ins.walk_deviation_pct) <= 10:
                parts_cn.append(f"今天走路量({ins.today_walk_min:.0f}分钟)和平时差不多")
                parts_en.append(f"Today's walking ({ins.today_walk_min:.0f}min) is about the same as usual")
            elif ins.walk_deviation_pct < -10:
                parts_cn.append(f"今天走路{ins.today_walk_min:.0f}分钟, 比平时少了{abs(ins.walk_deviation_pct):.0f}%")
                parts_en.append(f"Today walked {ins.today_walk_min:.0f}min, {abs(ins.walk_deviation_pct):.0f}% less than usual")
            else:
                parts_cn.append(f"今天走路{ins.today_walk_min:.0f}分钟, 比平时多了{ins.walk_deviation_pct:.0f}%")
                parts_en.append(f"Today walked {ins.today_walk_min:.0f}min, {ins.walk_deviation_pct:.0f}% more than usual")

        # 行为分布反常度
        if ins.behavior_divergence > 0.15:
            parts_cn.append("今天的行为节奏和平时明显不一样")
            parts_en.append("Today's activity rhythm differs notably from usual")
        elif ins.behavior_divergence > 0.05:
            parts_cn.append("今天行为略有变化")
            parts_en.append("Today's activity changed slightly")

        # 活动高峰
        if ins.today_peak_hour >= 0 and ins.usual_peak_hour >= 0:
            if ins.today_peak_hour != ins.usual_peak_hour:
                parts_cn.append(f"活动高峰从平时{ins.usual_peak_hour}点移到{ins.today_peak_hour}点")
                parts_en.append(f"Peak activity shifted from {ins.usual_peak_hour}h to {ins.today_peak_hour}h")

        # 风险事件
        if ins.high_risk_events > 0:
            parts_cn.append(f"今天有{ins.high_risk_events}次高风险波动")
            parts_en.append(f"{ins.high_risk_events} high-risk events today")

        if not parts_cn:
            parts_cn.append("今天和平时基本一致, 一切正常")
            parts_en.append("Today is broadly consistent with usual. All good.")
        return "；".join(parts_cn), " ".join(parts_en)

    # ════════════════════════════════════════════════════
    # 持久化
    # ════════════════════════════════════════════════════

    def save(self, path: Optional[str] = None) -> str:
        """保存画像. 归档当前天."""
        if self._today is not None and self._today_date:
            self.days[self._today_date] = self._today
        if path is None:
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, f"behavior_{self.person_id}.json")
        data = {
            "person_id": self.person_id,
            "fps": self.fps,
            "bucket_seconds": self.bucket_seconds,
            "days": {k: v.to_dict() for k, v in self.days.items()},
            "saved_at": time.time(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, cls=_SafeEncoder)
        return path

    @classmethod
    def load(cls, path: str) -> Optional["BehaviorProfile"]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        bp = cls(person_id=d["person_id"], fps=d.get("fps", 30.0),
                 bucket_seconds=d.get("bucket_seconds", 1800))
        for date_str, dbd in d.get("days", {}).items():
            bp.days[date_str] = DailyBehavior.from_dict(dbd)
        return bp

    @classmethod
    def load_or_create(cls, person_id: str, save_dir: str = "behavior_profiles",
                       fps: float = 30.0) -> "BehaviorProfile":
        path = os.path.join(save_dir, f"behavior_{person_id}.json")
        if os.path.exists(path):
            bp = cls.load(path)
            if bp is not None:
                bp.save_dir = save_dir
                bp.fps = fps if fps > 0 else bp.fps
                return bp
        return cls(person_id=person_id, save_dir=save_dir, fps=fps)


# ════════════════════════════════════════════════════
# 自测
# ════════════════════════════════════════════════════

def _test():
    print("=" * 60)
    print("行为画像 自测")
    print("=" * 60)

    import numpy as np

    bp = BehaviorProfile(person_id="T_elder", fps=30.0)
    base_ts = time.time()

    # ── 阶段1: 积累过去6天"平时"画像 (走量为多 + 一些坐下) ──
    print("\n[阶段1] 积累6天平时数据...")
    np.random.seed(7)
    # 今天0点
    today_date0 = time.strftime("%Y-%m-%d")
    today_start = datetime.strptime(today_date0, "%Y-%m-%d").timestamp()
    for day in range(6):
        # 严格用"今天之前"的整日 (day_start = today_start - (day+1)*86400)
        day_start = today_start - (day + 1) * 86400
        for i in range(3000):
            # 每天: 40%走路, 30%站立, 30%坐下 → 走路量比较足
            r = np.random.random()
            cid = 3 if r < 0.40 else (5 if r < 0.70 else 1)
            # 分散在一天 (早8点-晚20点), 保证不跨天
            hour = 8 + (i % 720) // 60
            ts = day_start + hour * 3600 + (i % 60) * 60
            bp.update(cid, ts)

    # 跨天切换 (让 profile 归档) → 确保今天从独立的一天开始
    bp._get_today(time.time())

    # ── 阶段2: 今天 (走量大幅减少: 只5%走路, 大量久坐) ──
    print("\n[阶段2] 模拟今天 (走路大幅减少)...")
    for i in range(3000):
        r = np.random.random()
        cid = 3 if r < 0.05 else (5 if r < 0.20 else 1)  # 走路仅5%
        # 今天也分散在 8点-20点
        hour = 8 + (i % 720) // 60
        ts = today_start + hour * 3600 + (i % 60) * 60
        bp.update(cid, ts)
    # 记录一个高风险事件
    bp.register_risk_event("DANGER", today_start + 12 * 3600)

    # ── 阶段3: 生成洞察 ──
    print("\n[阶段3] 生成今日洞察...")
    insight = bp.report_today(today_start + 20 * 3600)

    print(f"  今天日期: {insight.today_date}")
    print(f"  数据足够: {insight.has_enough_data}")
    print(f"  今天走路: {insight.today_walk_min:.1f}分钟 | 平时: {insight.usual_walk_min:.1f}分钟 | 偏差: {insight.walk_deviation_pct}%")
    print(f"  行为分布JS散度: {insight.behavior_divergence}")
    print(f"  活动高峰: 今天{insight.today_peak_hour}点 vs 平时{insight.usual_peak_hour}点")
    print(f"  风险事件: {insight.high_risk_events}")
    print(f"  摘要(CN): {insight.summary_cn}")
    print(f"  摘要(EN): {insight.summary_en}")

    assert insight.has_enough_data, "应有足够数据"
    assert insight.walk_deviation_pct is not None and insight.walk_deviation_pct < -10, \
        f"走路应大幅减少, 实际偏差 {insight.walk_deviation_pct}%"
    assert insight.high_risk_events == 1, f"应有1次风险, 实际 {insight.high_risk_events}"

    # ── 阶段4: 持久化 ──
    print("\n[阶段4] 持久化测试...")
    path = bp.save()
    print(f"  保存到: {path}")
    loaded = BehaviorProfile.load(path)
    assert loaded is not None
    assert loaded.person_id == "T_elder"
    assert len(loaded.days) >= 1
    print("  加载验证通过!")

    print("\n" + "=" * 60)
    print("行为画像 自测全部通过!")
    print("=" * 60)


if __name__ == "__main__":
    _test()
