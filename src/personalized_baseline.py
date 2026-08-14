#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
personalized_baseline.py — 个性化步态基线 v1.0
==============================================
创新层: "和过去的自己比, 而不是和人群平均比"

核心能力:
  1. 为每个老人建立个人步态指纹 (7项核心指标 + 分布)
  2. Z-score 偏离检测 → "偏离自己常态 2σ 以上" 触发关注
  3. 慢速 EMA 自适应 (基线随自然老化缓慢更新)
  4. 基线持久化 (跨会话, 重启不丢失)
  5. 多指标联合风险评分 + 临床显著性阈值

医学依据:
  - 步态速度是"第六生命体征" (Fritz & Lusardi, 2009)
  - 个人步态变异性比绝对值更能预测跌倒 (Hausdorff et al.)
  - 步速下降 >0.1 m/s = 有意义的临床变化 (Perera et al.)
  - 正常老化步速衰减 ≈ 0.2%/月 vs 病理衰减 > 1%/月

架构:
  PersonalBaseline (每人一个)
    ├── calibrate()       — 初始校准 (收集N帧数据)
    ├── update()          — 在线更新 + 偏离检测
    ├── score_deviation() — 多指标 Z-score → 个人风险分
    ├── save() / load()   — 持久化
    └── get_report()      — 可解释报告

集成点:
  e2e_fall_monitor.py → PersonState → 每人一个 PersonalBaseline
  gait_trend.py → TrendAnalyzer → 使用个人基线 (而非全局默认基线)
"""

import json
import os
import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy types to native Python types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════

@dataclass
class BaselineConfig:
    """个性化基线配置"""

    # ── 校准期 ──
    calibration_frames: int = 300          # 需要多少帧步行数据建立初始基线
    calibration_min_walking: int = 100     # 至少需要多少"行走"帧

    # ── 自适应速率 ──
    # EMA α: 每帧更新权重。值越小基线越稳定。
    #   0.0005 ≈ 原基线权重 99.95% / 新数据 0.05% (极慢, 数月才显著偏移)
    #   这模拟正常老化速度 (~0.2%/月)
    ema_alpha: float = 0.0005

    # ── 偏离阈值 (Z-score) ──
    z_warn: float = 1.5     # 轻微偏离 → 提醒
    z_alert: float = 2.5    # 显著偏离 → 告警
    z_danger: float = 3.5   # 极度偏离 → 高危

    # ── 临床显著性阈值 (绝对值, 即使Z-score低也有意义) ──
    clinical_speed_drop_ms: float = 0.10   # 步速下降>0.1 m/s (Perera 2006)
    clinical_sway_increase_deg: float = 3.0 # 摇摆增加>3° (Maki 1994)
    clinical_step_reduction_pct: float = 0.15 # 步长缩短>15%
    clinical_balance_drop: float = 0.10      # 平衡指数下降>0.10

    # ── 多指标联合评分权重 ──
    metric_weights: dict = field(default_factory=lambda: {
        "walking_speed":    0.30,  # 最重要的跌倒预测因子
        "step_length":      0.20,  # 步态模式改变
        "sway_angle":       0.20,  # 平衡控制
        "balance_index":    0.10,
        "knee_angle_avg":   0.05,
        "centroid_height":  0.05,
        "cadence":          0.10,
    })

    # ── 持久化 ──
    save_dir: str = "baselines"  # 基线保存目录


# 默认配置
DEFAULT_CONFIG = BaselineConfig()


# ════════════════════════════════════════════════════
# 单指标基线
# ════════════════════════════════════════════════════

class MetricBaseline:
    """
    单指标的个性化基线。
    维护该指标的均值 μ 和标准差 σ, 支持:
      - 校准期 (收集N个样本后建立初始 μ, σ)
      - EMA 自适应更新 (模拟自然老化)
      - Z-score 偏离检测
      - 临床显著性阈值
    """

    def __init__(self, name: str, ema_alpha: float = 0.0005,
                 clinical_threshold: Optional[float] = None,
                 direction: str = "bidirectional",
                 calib_min_samples: int = 50):
        """
        Args:
            name: 指标名称
            ema_alpha: 自适应更新速率
            clinical_threshold: 临床显著性阈值 (绝对值)
            direction: "both"=双向, "low_bad"=过低危险, "high_bad"=过高危险
            calib_min_samples: 校准所需最小样本数
        """
        self.name = name
        self.ema_alpha = ema_alpha
        self.clinical_threshold = clinical_threshold
        self.direction = direction
        self.calib_min_samples = calib_min_samples

        # 基线参数
        self.mu: float = 0.0       # 均值
        self.sigma: float = 0.0   # 标准差
        self.n_samples: int = 0   # 参与基线计算的样本数

        # 校准期收集
        self._calib_buffer: List[float] = []
        self._calibrated: bool = False

        # 最近值 (用于趋势)
        self._recent: deque = deque(maxlen=100)
        self._last_value: float = 0.0

    def calibrate(self, values: List[float]) -> bool:
        """用一批历史值建立初始基线"""
        if len(values) < 3:
            return False
        arr = np.array(values, dtype=np.float64)
        self.mu = float(np.mean(arr))
        self.sigma = max(float(np.std(arr)), 1e-6)
        self.n_samples = len(values)
        self._calibrated = True
        self._calib_buffer = []
        return True

    def add_calibration(self, value: float) -> bool:
        """收集校准样本, 收集够后自动建立基线"""
        self._calib_buffer.append(value)
        if len(self._calib_buffer) >= self.calib_min_samples:
            return self.calibrate(self._calib_buffer)
        return False

    def update(self, value: float) -> Dict[str, float]:
        """
        用新值更新基线并返回偏离度。
        返回: {"z_score": z, "clinical_flag": bool, "deviation_pct": pct}
        """
        self._last_value = value
        self._recent.append(value)

        if not self._calibrated:
            return {"z_score": 0.0, "clinical_flag": False, "deviation_pct": 0.0}

        # Z-score
        z = (value - self.mu) / self.sigma if self.sigma > 1e-9 else 0.0

        # 根据方向调整Z-score符号
        if self.direction == "low_bad":
            z = -z  # 值越低越危险 → 正Z
        elif self.direction == "high_bad":
            pass  # 值越高越危险 → 正Z

        # 偏离百分比
        if self.mu != 0:
            pct = (value - self.mu) / abs(self.mu)
        else:
            pct = 0.0

        # 临床显著性
        clinical_flag = False
        if self.clinical_threshold is not None:
            deviation = abs(value - self.mu)
            clinical_flag = deviation >= self.clinical_threshold

        # EMA 更新基线
        self.mu = float(self.ema_alpha * value + (1 - self.ema_alpha) * self.mu)
        # σ 也缓慢更新 (使用 MAD 的近似)
        if self.n_samples > 10:
            recent_std = max(float(np.std(list(self._recent))), 1e-6)
            self.sigma = float(self.ema_alpha * recent_std + (1 - self.ema_alpha) * self.sigma)
        self.n_samples += 1

        return {
            "z_score": round(z, 3),
            "clinical_flag": clinical_flag,
            "deviation_pct": round(pct, 4),
        }

    def get_z_score(self, value: Optional[float] = None) -> float:
        """获取当前或指定值的 Z-score"""
        v = value if value is not None else self._last_value
        if not self._calibrated or self.sigma < 1e-9:
            return 0.0
        z = (v - self.mu) / self.sigma
        if self.direction == "low_bad":
            z = -z
        return float(z)

    def to_dict(self) -> dict:
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "n_samples": self.n_samples,
            "calibrated": self._calibrated,
        }

    @classmethod
    def from_dict(cls, d: dict, name: str, ema_alpha: float = 0.0005,
                  clinical_threshold: Optional[float] = None,
                  direction: str = "bidirectional",
                  calib_min_samples: int = 50) -> "MetricBaseline":
        mb = cls(name=name, ema_alpha=ema_alpha,
                 clinical_threshold=clinical_threshold, direction=direction,
                 calib_min_samples=calib_min_samples)
        mb.mu = d["mu"]
        mb.sigma = d["sigma"]
        mb.n_samples = d["n_samples"]
        mb._calibrated = d.get("calibrated", True)
        return mb


# ════════════════════════════════════════════════════
# 个性化基线 (多指标)
# ════════════════════════════════════════════════════

@dataclass
class DeviationReport:
    """一次偏离检测报告"""
    person_id: str = ""
    calibrated: bool = False

    # ── 个人风险分 (0-100) ──
    personal_risk_score: float = 0.0
    alert_level: str = "SAFE"         # SAFE / NOTICE / WARN / ALERT / DANGER

    # ── 各指标偏离详情 ──
    deviations: Dict[str, float] = field(default_factory=dict)  # 指标 → Z-score
    clinical_flags: List[str] = field(default_factory=list)     # 触发的临床阈值

    # ── 当前值 vs 基线对比 ──
    current_metrics: dict = field(default_factory=dict)
    baseline_metrics: dict = field(default_factory=dict)

    # ── 趋势 ──
    trend_direction: str = "stable"   # stable / declining / improving
    days_since_baseline: float = 0.0

    # ── 建议 ──
    suggestion: str = ""
    timestamp: float = 0.0


class PersonalBaseline:
    """
    一个人的完整步态基线。

    用法:
        baseline = PersonalBaseline(person_id="elder_001")
        baseline.update(sample)       # 在线检测
        report = baseline.check()     # 获取偏离报告
        baseline.save()               # 持久化
    """

    # ── 指标定义: (名称, 方向, 临床阈值, 是否必需行走状态) ──
    METRICS = [
        ("walking_speed",    "low_bad",  0.10,  True),   # 步速 (归一化)
        ("step_length",      "low_bad",  0.15,  True),   # 步长 (比率)
        ("sway_angle",       "high_bad", 3.0,   False),  # 摇摆角 (°)
        ("balance_index",    "low_bad",  0.10,  False),  # 平衡指数
        ("knee_angle_avg",   "low_bad",  None,  True),   # 膝角度平均
        ("centroid_height",  "low_bad",  None,  False),  # 质心高度
        ("cadence",          "low_bad",  None,  True),   # 步频
        ("torso_angle",      "high_bad", None,  False),  # 躯干角
    ]

    def __init__(self, person_id: str, config: BaselineConfig = None):
        self.person_id = person_id
        self.cfg = config or DEFAULT_CONFIG

        # 为每个指标创建基线
        self.metrics: Dict[str, MetricBaseline] = {}
        for name, direction, clin_thresh, _ in self.METRICS:
            self.metrics[name] = MetricBaseline(
                name=name,
                ema_alpha=self.cfg.ema_alpha,
                clinical_threshold=clin_thresh,
                direction=direction,
                calib_min_samples=max(10, self.cfg.calibration_min_walking),
            )

        self._frame_count: int = 0
        self._walking_count: int = 0
        self._calibrated: bool = False
        self._calibration_start: float = time.time()
        self._baseline_established_at: Optional[float] = None

        # 历史报告 (用于趋势)
        self._recent_scores: deque = deque(maxlen=50)

    # ════════════════════════════════════════════════
    # 主接口
    # ════════════════════════════════════════════════

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def calibration_progress(self) -> float:
        """校准进度 0.0-1.0"""
        if self._calibrated:
            return 1.0
        return min(1.0, self._walking_count / max(self.cfg.calibration_min_walking, 1))

    def update(self, sample: 'GaitSample') -> Optional[DeviationReport]:
        """
        用一帧步态数据更新基线。
        在未校准时收集数据, 校准后检测偏离。
        """
        self._frame_count += 1

        # 提取指标值
        values = {
            "walking_speed":    sample.walking_speed,
            "step_length":      sample.step_length,
            "sway_angle":       sample.sway_angle,
            "balance_index":    sample.balance_index,
            "knee_angle_avg":   sample.knee_angle_avg,
            "centroid_height":  sample.centroid_height,
            "cadence":          0.0,  # 由外部提供
            "torso_angle":      sample.torso_angle,
        }

        # ── 校准阶段 ──
        if not self._calibrated:
            # 优先用走路数据校准
            if sample.is_walking:
                self._walking_count += 1
                # 所有指标都收集 (即使非走路也收集平衡相关)
                for name, _, _, needs_walking in self.METRICS:
                    if needs_walking and not sample.is_walking:
                        continue
                    val = values.get(name, 0.0)
                    if val != 0.0 or name in ("centroid_height", "torso_angle"):
                        self.metrics[name].add_calibration(val)

            # 检查是否收集够了
            if self._walking_count >= self.cfg.calibration_min_walking:
                self._finish_calibration()
            return None

        # ── 在线检测阶段 ──
        return self._check(values, sample.is_walking)

    def force_calibrate(self, historical_samples: List['GaitSample']) -> bool:
        """用历史数据强制校准"""
        walking = [s for s in historical_samples if s.is_walking]
        if len(walking) < 5:
            return False

        metric_values: Dict[str, List[float]] = {name: [] for name, _, _, _ in self.METRICS}
        for s in walking:
            vals = {
                "walking_speed": s.walking_speed,
                "step_length": s.step_length,
                "sway_angle": s.sway_angle,
                "balance_index": s.balance_index,
                "knee_angle_avg": s.knee_angle_avg,
                "centroid_height": s.centroid_height,
                "cadence": 0.0,
                "torso_angle": s.torso_angle,
            }
            for name in metric_values:
                val = vals.get(name, 0.0)
                if val != 0.0 or name in ("centroid_height", "torso_angle"):
                    metric_values[name].append(val)

        for name, values in metric_values.items():
            if len(values) >= 3:
                self.metrics[name].calibrate(values)

        self._calibrated = True
        self._baseline_established_at = time.time()
        self._walking_count = len(walking)
        return True

    def check(self) -> DeviationReport:
        """获取当前偏离报告 (不更新)"""
        return self._build_report()

    # ════════════════════════════════════════════════
    # 内部方法
    # ════════════════════════════════════════════════

    def _finish_calibration(self) -> None:
        """完成校准"""
        for name, _, _, _ in self.METRICS:
            mb = self.metrics[name]
            if not mb._calibrated and len(mb._calib_buffer) >= 3:
                mb.calibrate(mb._calib_buffer)

        self._calibrated = True
        self._baseline_established_at = time.time()

    def _check(self, values: Dict[str, float], is_walking: bool) -> DeviationReport:
        """执行偏离检测"""
        deviations = {}
        clinical_flags = []
        metric_risks = {}

        for name, direction, clin_thresh, needs_walking in self.METRICS:
            if needs_walking and not is_walking:
                continue

            val = values.get(name, 0.0)
            if val == 0.0 and name not in ("centroid_height", "torso_angle"):
                continue

            result = self.metrics[name].update(val)
            z = result["z_score"]
            deviations[name] = z

            if result["clinical_flag"]:
                clinical_flags.append(f"{name}: 超过临床阈值")

            # Z-score → 单指标风险 (0-100)
            abs_z = abs(z)
            if abs_z >= self.cfg.z_danger:
                metric_risks[name] = 80.0
            elif abs_z >= self.cfg.z_alert:
                metric_risks[name] = 60.0
            elif abs_z >= self.cfg.z_warn:
                metric_risks[name] = 35.0
            else:
                metric_risks[name] = max(0.0, abs_z / self.cfg.z_warn * 20.0)

        # ── 加权综合风险分 ──
        total_weight = 0.0
        weighted_sum = 0.0
        for name, risk in metric_risks.items():
            w = self.cfg.metric_weights.get(name, 0.05)
            weighted_sum += risk * w
            total_weight += w

        personal_risk = min(100.0, weighted_sum / max(total_weight, 0.01))

        # ── 临床附加分 (即使Z-score不高, 绝对值变化显著也加) ──
        if clinical_flags:
            personal_risk = min(100.0, personal_risk + 10.0 * len(clinical_flags))

        # ── 趋势 ──
        self._recent_scores.append(personal_risk)
        trend = self._compute_trend()

        # ── 告警等级 ──
        alert_level = self._classify_alert(personal_risk, len(clinical_flags), trend)

        # ── 建议 ──
        suggestion = self._generate_suggestion(alert_level, clinical_flags, deviations, trend)

        # ── 构建当前指标 vs 基线对比 ──
        current = {name: values.get(name, 0.0) for name, _, _, _ in self.METRICS}
        baseline = {name: self.metrics[name].mu for name in self.metrics}

        return DeviationReport(
            person_id=self.person_id,
            calibrated=True,
            personal_risk_score=round(personal_risk, 1),
            alert_level=alert_level,
            deviations={k: round(v, 2) for k, v in deviations.items()},
            clinical_flags=clinical_flags,
            current_metrics=current,
            baseline_metrics=baseline,
            trend_direction=trend,
            days_since_baseline=(time.time() - (self._baseline_established_at or time.time())) / 86400,
            suggestion=suggestion,
            timestamp=time.time(),
        )

    def _build_report(self) -> DeviationReport:
        """构建当前状态报告 (不更新基线)"""
        current = {}
        baseline = {}
        deviations = {}

        for name, _, _, _ in self.METRICS:
            mb = self.metrics[name]
            current[name] = mb._last_value
            baseline[name] = mb.mu
            deviations[name] = mb.get_z_score()

        trend = self._compute_trend()
        clinical_flags = []  # 历史快照不重复计算

        return DeviationReport(
            person_id=self.person_id,
            calibrated=self._calibrated,
            personal_risk_score=round(float(np.mean(list(self._recent_scores))) if self._recent_scores else 0, 1),
            alert_level="SAFE",
            deviations=deviations,
            clinical_flags=clinical_flags,
            current_metrics=current,
            baseline_metrics=baseline,
            trend_direction=trend,
            days_since_baseline=(time.time() - (self._baseline_established_at or time.time())) / 86400,
            suggestion="",
            timestamp=time.time(),
        )

    def _compute_trend(self) -> str:
        """计算近期风险趋势"""
        if len(self._recent_scores) < 10:
            return "stable"
        recent = list(self._recent_scores)[-20:]
        if len(recent) < 10:
            return "stable"
        half = len(recent) // 2
        first = np.mean(recent[:half])
        second = np.mean(recent[-half:])
        if second - first > 5:
            return "declining"
        elif first - second > 5:
            return "improving"
        return "stable"

    def _classify_alert(self, risk: float, n_clinical: int, trend: str) -> str:
        """个人基线告警等级"""
        if trend == "declining":
            risk *= 1.15  # 恶化趋势加成

        if risk >= 75 or n_clinical >= 3:
            return "DANGER"
        elif risk >= 55 or n_clinical >= 2:
            return "ALERT"
        elif risk >= 35 or n_clinical >= 1:
            return "WARN"
        elif risk >= 15:
            return "NOTICE"
        return "SAFE"

    def _generate_suggestion(self, level: str, clinical_flags: List[str],
                             deviations: Dict[str, float], trend: str) -> str:
        """生成可读建议"""
        if level == "SAFE":
            return "步态正常，与个人基线一致"

        parts = []

        # 找出最异常的指标
        sorted_devs = sorted(deviations.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, z in sorted_devs[:3]:
            if abs(z) < self.cfg.z_warn:
                break
            direction = "下降" if z > 0 else "升高"
            if self.metrics[name].direction == "high_bad" and z < 0:
                direction = "改善"
            elif self.metrics[name].direction == "low_bad" and z > 0:
                direction = "降低"

            label_map = {
                "walking_speed": "步速", "step_length": "步长",
                "sway_angle": "躯干摇摆", "balance_index": "平衡",
                "knee_angle_avg": "膝活动度", "centroid_height": "身体高度",
                "cadence": "步频", "torso_angle": "躯干倾斜",
            }
            label = label_map.get(name, name)
            parts.append(f"{label}异常{direction}(Z={abs(z):.1f})")

        if clinical_flags:
            parts.append(f"{len(clinical_flags)}项指标达临床显著性阈值")

        if trend == "declining":
            parts.append("近期趋势持续恶化")
        elif trend == "improving":
            parts.append("近期趋势正在改善")

        base = "；".join(parts)

        if level == "DANGER":
            return f"[!!] 高度偏离个人基线！{base}。建议立即评估跌倒风险"
        elif level == "ALERT":
            return f"[!!] 显著偏离个人基线。{base}。建议关注并评估"
        elif level == "WARN":
            return f"[!] 轻微偏离个人基线。{base}。建议持续观察"
        elif level == "NOTICE":
            return f"[i] 步态有微小变化。{base}"
        return ""

    # ════════════════════════════════════════════════
    # 持久化
    # ════════════════════════════════════════════════

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "calibrated": self._calibrated,
            "frame_count": self._frame_count,
            "walking_count": self._walking_count,
            "baseline_established_at": self._baseline_established_at,
            "created_at": time.time(),
            "metrics": {name: mb.to_dict() for name, mb in self.metrics.items()},
        }

    def save(self, path: Optional[str] = None) -> str:
        """保存基线到文件, 返回文件路径"""
        if path is None:
            os.makedirs(self.cfg.save_dir, exist_ok=True)
            path = os.path.join(self.cfg.save_dir, f"baseline_{self.person_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, cls=_SafeEncoder)
        return path

    @classmethod
    def load(cls, path: str, config: BaselineConfig = None) -> Optional["PersonalBaseline"]:
        """从文件加载基线"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        cfg = config or DEFAULT_CONFIG
        pb = cls(person_id=d["person_id"], config=cfg)
        pb._calibrated = d.get("calibrated", False)
        pb._frame_count = d.get("frame_count", 0)
        pb._walking_count = d.get("walking_count", 0)
        pb._baseline_established_at = d.get("baseline_established_at")

        # 从 METRICS 构建名称→ (direction, clinical_threshold) 映射
        metrics_lookup = {m[0]: (m[1], m[2]) for m in pb.METRICS}

        for name, mb_dict in d.get("metrics", {}).items():
            if name in pb.metrics:
                direction, clin_thresh = metrics_lookup.get(name, ("bidirectional", None))
                pb.metrics[name] = MetricBaseline.from_dict(
                    mb_dict, name=name, ema_alpha=cfg.ema_alpha,
                    clinical_threshold=clin_thresh, direction=direction,
                    calib_min_samples=max(10, cfg.calibration_min_walking))

        return pb

    @classmethod
    def load_or_create(cls, person_id: str, config: BaselineConfig = None,
                       save_dir: str = None) -> "PersonalBaseline":
        """加载已有基线, 不存在则创建新的"""
        cfg = config or DEFAULT_CONFIG
        d = save_dir or cfg.save_dir
        path = os.path.join(d, f"baseline_{person_id}.json")

        if os.path.exists(path):
            pb = cls.load(path, config=cfg)
            if pb is not None:
                return pb

        return cls(person_id=person_id, config=cfg)

    # ════════════════════════════════════════════════
    # 摘要
    # ════════════════════════════════════════════════

    def summary(self) -> str:
        """人类可读的基线摘要"""
        if not self._calibrated:
            progress = self.calibration_progress
            return (f"[{self.person_id}] 校准中... ({progress*100:.0f}%, "
                    f"{self._walking_count}/{self.cfg.calibration_min_walking} 行走帧)")

        lines = [f"[{self.person_id}] 个性化步态基线 (已校准 {self._frame_count} 帧)"]
        label_map = {
            "walking_speed": "步速", "step_length": "步长",
            "sway_angle": "摇摆角", "balance_index": "平衡指数",
            "knee_angle_avg": "膝角度", "centroid_height": "质心高",
            "cadence": "步频", "torso_angle": "躯干角",
        }
        for name, mb in self.metrics.items():
            if mb._calibrated:
                label = label_map.get(name, name)
                lines.append(f"  {label:8s}: μ={mb.mu:.4f}  σ={mb.sigma:.4f}  (n={mb.n_samples})")

        # 最近风险
        if self._recent_scores:
            avg_risk = np.mean(list(self._recent_scores)[-10:])
            lines.append(f"  近期平均个人风险: {avg_risk:.1f}/100")

        return "\n".join(lines)


# ════════════════════════════════════════════════════
# 集成工具: 将 GaitSample 桥接到 PersonalBaseline
# ════════════════════════════════════════════════════

def create_baseline_from_gait_samples(
    person_id: str,
    samples: List['GaitSample'],
    config: BaselineConfig = None,
) -> PersonalBaseline:
    """从历史步态样本创建并校准基线"""
    baseline = PersonalBaseline(person_id=person_id, config=config)
    baseline.force_calibrate(samples)
    return baseline


# ════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════

def _test():
    """模拟测试: 建立基线 → 模拟恶化 → 检测偏离"""
    print("=" * 60)
    print("个性化基线模块 自测")
    print("=" * 60)

    from src.gait_trend import GaitSample

    cfg = BaselineConfig(calibration_frames=50, calibration_min_walking=30)
    baseline = PersonalBaseline(person_id="test_elder", config=cfg)

    # ── 阶段1: 校准 (模拟一个老人的正常步态) ──
    print("\n[阶段1] 建立个人基线 (模拟正常行走)...")
    np.random.seed(42)
    calib_samples = []
    for i in range(100):
        s = GaitSample(
            timestamp=time.time() - (100 - i) * 600,
            step_length=0.15 + np.random.normal(0, 0.008),
            walking_speed=0.05 + np.random.normal(0, 0.003),
            sway_angle=3.0 + np.random.normal(0, 0.5),
            balance_index=0.92 + np.random.normal(0, 0.02),
            torso_angle=2.0 + np.random.normal(0, 0.3),
            centroid_height=0.35 + np.random.normal(0, 0.01),
            knee_angle_avg=168.0 + np.random.normal(0, 3.0),
            is_walking=True,
        )
        calib_samples.append(s)
        baseline.update(s)

    print(f"  校准完成: {baseline.is_calibrated}")
    print(baseline.summary())

    # ── 阶段2: 模拟步态恶化 (步速下降20%, 摇摆增加50%) ──
    print("\n[阶段2] 模拟步态逐渐恶化...")
    reports = []
    for i in range(50):
        deterioration = i / 50.0  # 0 → 1 线性恶化
        s = GaitSample(
            timestamp=time.time(),
            step_length=0.15 * (1 - deterioration * 0.25) + np.random.normal(0, 0.01),
            walking_speed=0.05 * (1 - deterioration * 0.30) + np.random.normal(0, 0.005),
            sway_angle=3.0 * (1 + deterioration * 0.60) + np.random.normal(0, 0.8),
            balance_index=0.92 * (1 - deterioration * 0.15) + np.random.normal(0, 0.03),
            torso_angle=2.0 + deterioration * 4.0 + np.random.normal(0, 0.5),
            centroid_height=0.35 - deterioration * 0.03 + np.random.normal(0, 0.01),
            knee_angle_avg=168.0 * (1 - deterioration * 0.12) + np.random.normal(0, 3.0),
            is_walking=True,
        )
        report = baseline.update(s)
        if report:
            reports.append(report)

    if reports:
        last = reports[-1]
        print(f"\n  最终个人风险分: {last.personal_risk_score:.1f}/100")
        print(f"  告警等级: {last.alert_level}")
        print(f"  各指标 Z-score 偏离:")
        for name, z in sorted(last.deviations.items(), key=lambda x: abs(x[1]), reverse=True):
            bar = "█" * min(int(abs(z) * 5), 30)
            print(f"    {name:20s}: Z={z:+.2f} {bar}")
        if last.clinical_flags:
            print(f"  临床阈值触发: {last.clinical_flags}")
        if last.suggestion:
          print(f"  建议: {last.suggestion}")

        # 验证: 恶化场景应该有风险分 > 0
        assert last.personal_risk_score > 20, f"期望风险分>20, 实际={last.personal_risk_score:.1f}"

    # ── 持久化测试 ──
    print("\n[阶段3] 持久化测试...")
    path = baseline.save()
    print(f"  保存到: {path}")
    loaded = PersonalBaseline.load(path)
    assert loaded is not None
    assert loaded.person_id == "test_elder"
    assert loaded.is_calibrated
    print("  加载验证通过!")

    print("\n" + "=" * 60)
    print("个性化基线模块 自测全部通过!")
    print("=" * 60)


if __name__ == "__main__":
    _test()
