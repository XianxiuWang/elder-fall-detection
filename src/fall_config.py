#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fall_config.py — 跌倒检测系统统一配置
=====================================
所有可调参数集中管理，支持 JSON 导入导出，方便调参和自动化优化。

用法:
    from .fall_config import FallConfig
    cfg = FallConfig()                    # 默认配置
    cfg = FallConfig.load("params.json")  # 从文件加载
    cfg.save("params.json")              # 保存到文件
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple
import json
import os


# ============================================================
# 子配置
# ============================================================

@dataclass
class MediaPipeConfig:
    """MediaPipe Pose 配置"""
    model_complexity: int = 1          # 1=轻量, 2=高精（需下载）
    min_detection_confidence: float = 0.3
    min_tracking_confidence: float = 0.3
    static_image_mode: bool = False
    smooth_landmarks: bool = True


@dataclass
class MotionSpatialConfig:
    """空间运动分析器参数"""
    motion_threshold: float = 0.005        # 判定关键点"在运动"的归一化位移阈值
    spread_smooth: int = 5                 # 运动扩散范围的平滑窗口帧数

    # is_likely_fall() 判定阈值
    fall_centroid_disp_min: float = 0.02   # 运动中心位移最低阈值
    fall_centroid_disp_max: float = 0.08   # 运动中心位移饱和阈值
    fall_spread_active_min: float = 0.4    # 运动关键点比例最低阈值
    fall_spread_active_max: float = 0.8    # 运动关键点比例饱和阈值
    fall_spread_width_min: float = 0.2     # 空间跨度最低阈值
    fall_spread_width_max: float = 0.6     # 空间跨度饱和阈值
    fall_torso_ratio_min: float = 0.3      # 躯干位移占比最低阈值

    # 判定分数权重（is_likely_fall）
    likelihood_weights: Dict[str, float] = field(default_factory=lambda: {
        "centroid_disp": 1.0,
        "spread_active": 1.0,
        "spread_width": 1.0,
        "torso_ratio": 1.0,
    })
    likelihood_confidence_threshold: float = 0.5  # 判定为跌倒的置信度阈值


@dataclass
class ProcessFallConfig:
    """过程级跌倒检测器参数"""
    window_seconds: float = 2.0         # 分析窗口时间
    frame_rate: float = 30.0            # 摄像头帧率
    min_sequence_frames: int = 15       # 触发收集的最少序列帧数
    fall_height_threshold: float = 0.08 # 质心下降阈值（归一化）
    fall_speed_threshold: float = 0.03  # 运动中心速度阈值（归一化）
    cooldown_seconds: float = 5.0       # 报警冷却时间
    consecutive_suspicious: int = 3     # 连续触发帧数才进入收集模式

    # 单帧疑似判定阈值
    suspicious_centroid_disp: float = 0.03
    suspicious_motion_spread: float = 0.3
    suspicious_torso_disp_ratio: float = 0.25
    suspicious_torso_angle: float = 30.0  # 度
    suspicious_min_signals: int = 3

    # 单帧风险分计算
    risk_centroid_disp_mult: float = 20.0  # 速度→风险 乘数
    risk_centroid_disp_weight: float = 0.35
    risk_spread_weight: float = 0.30
    risk_torso_disp_weight: float = 0.20
    risk_torso_angle_weight: float = 0.15
    risk_torso_angle_max: float = 90.0

    # 5维分析权重
    analysis_weights: Dict[str, float] = field(default_factory=lambda: {
        "speed": 0.25,
        "height": 0.30,
        "torso": 0.20,
        "spread": 0.10,
        "stillness": 0.15,
    })

    # 告警等级阈值
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "SUSPICIOUS": 0.4,
        "WARNING": 0.6,
        "ALERT": 0.8,
        "URGENT": 0.95,
    })

    # 速度轨迹子参数
    speed_peak_mean_ratio_high: float = 3.0     # 峰/均比高分阈值
    speed_peak_mean_ratio_mid: float = 2.0      # 峰/均比中分阈值
    speed_peak_mean_ratio_low: float = 1.5      # 峰/均比低分阈值
    speed_deceleration_high: float = 0.3        # 减速比高分阈值
    speed_deceleration_low: float = 0.1         # 减速比低分阈值

    # 高度轨迹子参数
    height_drop_2x_score: float = 1.0           # 下降量 2x 阈值时满分
    height_drop_1x_score: float = 0.6           # 下降量 1x 阈值时分数
    height_drop_03x_score: float = 0.2          # 下降量 0.3x 阈值时分数
    height_not_recovered_weight: float = 0.2
    height_late_peak_weight: float = 0.2
    height_drop_weight: float = 0.6

    # 躯干姿态轨迹子参数
    torso_final_angle_high: float = 60.0        # 最终角度高分阈值
    torso_final_angle_mid: float = 40.0
    torso_final_angle_low: float = 25.0
    torso_change_high: float = 40.0             # 变化幅度高分阈值
    torso_change_mid: float = 20.0
    torso_monotonicity_high: float = 0.8        # 单调性高分阈值
    torso_monotonicity_mid: float = 0.5
    torso_angle_weight: float = 0.4
    torso_change_weight: float = 0.3
    torso_mono_weight: float = 0.3

    # 运动扩散轨迹子参数
    spread_max_high: float = 0.6
    spread_max_mid: float = 0.4
    spread_max_low: float = 0.2
    spread_change_mult: float = 3.0
    spread_max_weight: float = 0.6
    spread_change_weight: float = 0.4

    # 后段静止子参数
    stillness_ratio_high: float = 0.2
    stillness_ratio_mid: float = 0.5
    stillness_ratio_low: float = 0.8
    stillness_tail_fraction: float = 1.0 / 3.0

    # 高度不够时降权
    low_height_drop_penalty: float = 0.3


@dataclass
class GaitTrendConfig:
    """步态趋势分析参数"""
    window_days: int = 7
    samples_per_day: int = 100
    min_samples_for_analysis: int = 50
    analysis_interval: int = 100          # 每 N 个样本重新分析一次
    baseline_history_days: int = 3

    # 趋势权重
    trending_weights: Dict[str, float] = field(default_factory=lambda: {
        "baseline": 0.30,
        "trend": 0.40,
        "instability": 0.30,
    })

    # 各步态指标权重
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "step_length": 0.25,
        "walking_speed": 0.25,
        "sway_angle": 0.20,
        "balance_index": 0.10,
        "torso_angle": 0.10,
        "centroid_height": 0.05,
        "knee_angle_avg": 0.05,
    })

    # 行走状态判定阈值
    walking_knee_min: float = 130.0
    walking_knee_max: float = 185.0
    walking_torso_max: float = 25.0
    walking_height_min: float = 0.20
    walking_height_max: float = 0.50
    walking_speed_min: float = 0.001

    # 基线默认值
    default_baseline: Dict[str, float] = field(default_factory=lambda: {
        "step_length": 0.15,
        "walking_speed": 0.05,
        "sway_angle": 5.0,
        "balance_index": 0.95,
        "torso_angle": 3.0,
        "centroid_height": 0.35,
        "knee_angle_avg": 170.0,
    })

    # 偏离基线评分
    max_deviation_ratio: float = 3.0  # 偏离比例上限

    # 不稳定性评分
    instability_cv_offset: float = 0.05   # 变异系数偏移
    instability_cv_mult: float = 200.0    # 变异系数→分数 乘数

    # 趋势斜率放大
    step_slope_mult: float = 100.0
    speed_slope_mult: float = 100.0
    sway_slope_mult: float = 50.0

    # 告警等级
    alert_levels: Dict[Tuple[float, float], str] = field(default_factory=lambda: {
        (0, 25): "green",
        (25, 50): "yellow",
        (50, 75): "orange",
        (75, 101): "red",
    })


@dataclass
class E2EMonitorConfig:
    """端到端监控配置"""
    rtsp_url: str = "rtsp://admin:RVXCEM@192.168.1.100:554/h264/ch1/main/av_stream"
    window_name: str = "E2E Fall Monitor | Q=Quit"
    alert_display_seconds: float = 3.0   # 告警横幅显示时间
    use_local_camera_fallback: bool = True
    camera_buffer_size: int = 2
    progress_log_interval: int = 30      # 每 N 帧打印进度

    # 久坐提醒 (v3.0)
    sedentary_warn_min: float = 45.0     # 预警: 连续坐多久提醒
    sedentary_alert_min: float = 60.0    # 告警: 连续坐多久强制提醒

    # HUD 颜色
    red: Tuple[int, int, int] = (0, 0, 255)
    green: Tuple[int, int, int] = (0, 255, 0)
    yellow: Tuple[int, int, int] = (0, 255, 255)
    orange: Tuple[int, int, int] = (0, 165, 255)
    white: Tuple[int, int, int] = (255, 255, 255)
    black: Tuple[int, int, int] = (0, 0, 0)
    gray: Tuple[int, int, int] = (128, 128, 128)
    cyan: Tuple[int, int, int] = (0, 200, 200)  # 低置信跟踪


# ============================================================
# 总配置
# ============================================================

@dataclass
class FallConfig:
    """跌倒检测系统总配置"""
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    motion_spatial: MotionSpatialConfig = field(default_factory=MotionSpatialConfig)
    process_fall: ProcessFallConfig = field(default_factory=ProcessFallConfig)
    gait_trend: GaitTrendConfig = field(default_factory=GaitTrendConfig)
    e2e_monitor: E2EMonitorConfig = field(default_factory=E2EMonitorConfig)

    # 元信息
    version: str = "2.0"
    description: str = ""

    def save(self, path: str) -> None:
        """保存配置到 JSON 文件"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        d = asdict(self)
        d = self._make_json_safe(d)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _make_json_safe(obj):
        """递归转换不可 JSON 序列化的类型。tuple → list，tuple key → string key"""
        if isinstance(obj, dict):
            return {FallConfig._json_key(k): FallConfig._make_json_safe(v)
                    for k, v in obj.items()}
        elif isinstance(obj, list):
            return [FallConfig._make_json_safe(v) for v in obj]
        elif isinstance(obj, tuple):
            # v2.1: tuple 值 → list (JSON 原生序列化)，不再转成字符串
            # 例: (0, 255, 0) → [0, 255, 0]
            return list(obj)
        else:
            return obj

    @staticmethod
    def _json_key(k):
        """将 dict key 转换为 JSON-safe 类型"""
        if isinstance(k, tuple):
            return f"({','.join(str(x) for x in k)})"
        return k

    @staticmethod
    def _parse_json_loaded(obj):
        """反向转换：处理历史遗留的 str 化 tuple（key + value）和新版 list→tuple"""
        import re
        # 匹配字符串化的元组: "(123, 456, 789)" 或 "(0, 255, 255)"
        _TUPLE_STR_RE = re.compile(r'^\(\s*-?\d+(?:\.\d+)?\s*(?:,\s*-?\d+(?:\.\d+)?\s*)+\)$')

        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                new_k = k
                # 还原 tuple key
                if isinstance(k, str) and k.startswith("(") and k.endswith(")"):
                    try:
                        parts = [float(x.strip()) for x in k[1:-1].split(",")]
                        if len(parts) == 2:
                            new_k = (parts[0], parts[1])
                    except (ValueError, IndexError):
                        pass
                new_v = FallConfig._parse_json_loaded(v)
                result[new_k] = new_v
            return result
        elif isinstance(obj, list):
            return [FallConfig._parse_json_loaded(v) for v in obj]
        elif isinstance(obj, str) and _TUPLE_STR_RE.match(obj):
            # v2.1: 还原历史遗留的字符串化 tuple (如 "(128, 128, 128)")
            try:
                parts = [int(x.strip()) for x in obj[1:-1].split(",")]
                return tuple(parts)
            except (ValueError, IndexError):
                return obj
        else:
            return obj

    @classmethod
    def load(cls, path: str) -> "FallConfig":
        """从 JSON 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        d = cls._parse_json_loaded(d)

        cfg = cls()
        if "mediapipe" in d:
            cfg.mediapipe = MediaPipeConfig(**d["mediapipe"])
        if "motion_spatial" in d:
            cfg.motion_spatial = MotionSpatialConfig(**d["motion_spatial"])
        if "process_fall" in d:
            cfg.process_fall = ProcessFallConfig(**d["process_fall"])
        if "gait_trend" in d:
            cfg.gait_trend = GaitTrendConfig(**d["gait_trend"])
        if "e2e_monitor" in d:
            cfg.e2e_monitor = E2EMonitorConfig(**d["e2e_monitor"])
        cfg.version = d.get("version", "2.0")
        cfg.description = d.get("description", "")
        return cfg

    def to_flat_dict(self) -> Dict[str, float]:
        """展开为扁平的 key→value 字典（供调参优化使用）"""
        flat = {}
        for section_name, section in [
            ("motion", self.motion_spatial),
            ("process", self.process_fall),
            ("gait", self.gait_trend),
        ]:
            for k, v in asdict(section).items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, (int, float)):
                            flat[f"{section_name}.{k}.{sub_k}"] = float(sub_v)
                elif isinstance(v, (int, float)):
                    flat[f"{section_name}.{k}"] = float(v)
        return flat

    def update_from_flat(self, flat: Dict[str, float]) -> None:
        """从扁平字典更新参数"""
        for key_path, value in flat.items():
            parts = key_path.split(".")
            section_name, attr = parts[0], parts[1]
            section = getattr(self, {
                "motion": "motion_spatial",
                "process": "process_fall",
                "gait": "gait_trend",
            }[section_name])

            if len(parts) == 2:
                if hasattr(section, attr):
                    setattr(section, attr, type(getattr(section, attr))(value))
            elif len(parts) == 3:
                obj = getattr(section, attr)
                if isinstance(obj, dict) and parts[2] in obj:
                    obj[parts[2]] = value

    def copy(self) -> "FallConfig":
        """深拷贝"""
        return FallConfig.load_from_dict(asdict(self))

    @classmethod
    def load_from_dict(cls, d: dict) -> "FallConfig":
        """从字典重建配置（供 copy 使用）"""
        cfg = cls()
        for section_name, section_cls, key in [
            ("mediapipe", MediaPipeConfig, "mediapipe"),
            ("motion_spatial", MotionSpatialConfig, "motion_spatial"),
            ("process_fall", ProcessFallConfig, "process_fall"),
            ("gait_trend", GaitTrendConfig, "gait_trend"),
            ("e2e_monitor", E2EMonitorConfig, "e2e_monitor"),
        ]:
            if key in d:
                setattr(cfg, key, section_cls(**d[key]))
        cfg.version = d.get("version", "2.0")
        cfg.description = d.get("description", "")
        return cfg


# ============================================================
# 快捷函数
# ============================================================

def load_or_default(path: str = None) -> FallConfig:
    """加载配置文件，不存在则返回默认配置"""
    if path and os.path.exists(path):
        return FallConfig.load(path)
    return FallConfig()


# 默认配置实例（可直接 import 使用）
DEFAULT_CONFIG = FallConfig()
