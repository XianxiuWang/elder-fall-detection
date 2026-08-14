#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
param_tuner.py — 跌倒识别参数自动调优
=====================================
Phase C: 用模拟数据（6类场景）做随机搜索 + 精炼，选出最优参数组合。

6 类场景:
  1. 行走 (WALKING)     — 不应告警
  2. 跌倒 (FALLING)     — 必须告警（高置信度优先）
  3. 弯腰捡物 (BENDING)  — 不应告警（易误报场景）
  4. 坐下 (SITTING)     — 不应告警
  5. 挥手 (WAVING)      — 不应告警
  6. 蹲下起立 (SQUATTING)— 边界场景，允许 SUSPICIOUS

搜索策略: 随机搜索 4000 组 → Top 20 精炼（小范围扰动）→ 输出最优配置

用法:
    conda activate fall
    python E:\老人跌倒\param_tuner.py
    python E:\老人跌倒\param_tuner.py --search 6000 --output best_params.json
"""

import argparse
import json
import os
import sys
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "src"))
import time
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np

sys.path.insert(0, r'E:\老人跌倒')

from fall_config import FallConfig, MotionSpatialConfig, ProcessFallConfig
from process_fall_detector import ProcessFallDetector, FallAlert

# ============================================================
# 场景生成器
# ============================================================

@dataclass
class ScenarioFrame:
    """一帧的模拟数据（process_fall 所需字段）"""
    centroid_y: float       # features[1]
    torso_angle: float      # features[3]
    centroid_disp: float    # spatial[0]
    motion_spread: float    # spatial[3]
    torso_disp_ratio: float # spatial[4]


def _add_noise(val: float, noise: float = 0.02) -> float:
    """高斯噪声"""
    return val + np.random.normal(0, noise)


def _walking_pattern(t: float, phase: float = 0) -> Tuple[float, float, float, float, float]:
    """行走模式：周期性摆动 (t 为帧序号)"""
    cycle = 2 * np.pi * t / 25  # ~25帧一个步态周期
    return (
        _add_noise(0.35 + 0.02 * np.sin(cycle + phase)),   # centroid_y: 微幅摆动
        _add_noise(5.0 + 3.0 * np.sin(cycle + phase)),     # torso_angle: 3-8°
        _add_noise(0.004 + 0.002 * abs(np.sin(cycle))),    # centroid_disp: 规则步幅
        _add_noise(0.10),                                    # motion_spread: 低散布
        _add_noise(0.28),                                    # torso_disp_ratio: 正常
    )


def generate_scenarios(seed: int = 42, num_variants: int = 15) -> dict:
    """
    生成 6 类场景，每类 num_variants 个变体（加入噪声）。
    返回 {scenario_name: List[List[ScenarioFrame]]}
    """
    np.random.seed(seed)
    random.seed(seed)
    scenarios: Dict[str, List[List[ScenarioFrame]]] = {}

    # ── 1. 行走 ──
    variants = []
    for v in range(num_variants):
        frames = []
        phase_shift = random.uniform(0, 2 * np.pi)
        for t in range(80):
            cy, ta, cd, ms, tr = _walking_pattern(t, phase_shift)
            frames.append(ScenarioFrame(cy, ta, cd, ms, tr))
        variants.append(frames)
    scenarios["walking"] = variants

    # ── 2. 跌倒 ──
    variants = []
    for v in range(num_variants):
        frames = []
        fall_start = random.randint(25, 35)     # 跌倒起始帧
        fall_duration = random.randint(15, 25)   # 跌倒持续帧数
        fall_speed = random.uniform(0.9, 1.1)    # 速度因子
        fall_angle = random.uniform(70, 90)      # 最终躯干角

        for t in range(90):
            if t < fall_start:
                cy, ta, cd, ms, tr = _walking_pattern(t, random.uniform(0, 2 * np.pi))
            elif t < fall_start + fall_duration:
                progress = (t - fall_start) / fall_duration
                cy = _add_noise(0.35 + progress * 0.35 * fall_speed)
                ta = _add_noise(5.0 + progress * fall_angle)
                cd = _add_noise(0.004 + progress * 0.06 * fall_speed)
                ms = _add_noise(0.10 + progress * 0.70)
                tr = _add_noise(0.28 + progress * 0.60)
            else:
                # 跌倒后静止
                settle = (t - fall_start - fall_duration) / 5
                settle = min(1.0, settle)
                cy = _add_noise(0.70 + settle * 0.02)
                ta = _add_noise(fall_angle + settle * 5)
                cd = _add_noise(0.0005)
                ms = _add_noise(0.04)
                tr = _add_noise(0.15)
            frames.append(ScenarioFrame(cy, ta, cd, ms, tr))
        variants.append(frames)
    scenarios["falling"] = variants

    # ── 3. 弯腰捡物 ──
    variants = []
    for v in range(num_variants):
        frames = []
        bend_start = random.randint(20, 30)
        bend_duration = random.randint(12, 18)
        bend_angle = random.uniform(40, 55)        # 弯腰角度（小于跌倒）
        bend_height = random.uniform(0.10, 0.18)   # 质心下降（小于跌倒）

        for t in range(80):
            if t < bend_start:
                cy, ta, cd, ms, tr = _walking_pattern(t)
            elif t < bend_start + bend_duration:
                progress = (t - bend_start) / bend_duration
                cy = _add_noise(0.35 + progress * bend_height)
                ta = _add_noise(5.0 + progress * bend_angle)
                cd = _add_noise(0.004 + progress * 0.015)
                ms = _add_noise(0.10 + progress * 0.15)
                tr = _add_noise(0.28 + progress * 0.25)
            elif t < bend_start + bend_duration * 2:
                # 恢复阶段
                progress = (t - bend_start - bend_duration) / bend_duration
                cy = _add_noise(0.35 + bend_height * (1 - progress))
                ta = _add_noise(bend_angle * (1 - progress) + 5.0 * progress)
                cd = _add_noise(0.004)
                ms = _add_noise(0.10 + 0.05 * (1 - progress))
                tr = _add_noise(0.28)
            else:
                cy, ta, cd, ms, tr = _walking_pattern(t)
            frames.append(ScenarioFrame(cy, ta, cd, ms, tr))
        variants.append(frames)
    scenarios["bending"] = variants

    # ── 4. 坐下 ──
    variants = []
    for v in range(num_variants):
        frames = []
        sit_start = random.randint(20, 30)
        sit_duration = random.randint(20, 30)

        for t in range(80):
            if t < sit_start:
                cy, ta, cd, ms, tr = _walking_pattern(t)
            elif t < sit_start + sit_duration:
                progress = (t - sit_start) / sit_duration
                cy = _add_noise(0.35 + progress * 0.22)     # 坐下质心降
                ta = _add_noise(5.0 - progress * 3.0)        # 躯干近直立（关键区别）
                cd = _add_noise(0.004 + progress * 0.008 * (1 - progress))  # 慢速
                ms = _add_noise(0.08 + progress * 0.05)
                tr = _add_noise(0.30 + progress * 0.10)
            else:
                cy = _add_noise(0.57)
                ta = _add_noise(2.0)
                cd = _add_noise(0.0005)
                ms = _add_noise(0.03)
                tr = _add_noise(0.15)
            frames.append(ScenarioFrame(cy, ta, cd, ms, tr))
        variants.append(frames)
    scenarios["sitting"] = variants

    # ── 5. 挥手 ──
    variants = []
    for v in range(num_variants):
        frames = []
        for t in range(50):
            # 身体不动，只有手臂动——但 process_fall 只看质心+躯干
            cy = _add_noise(0.35)
            ta = _add_noise(3.0)
            cd = _add_noise(0.0003)   # 近似不动
            ms = _add_noise(0.02)      # 身体 motion 很低
            tr = _add_noise(0.15)      # torso 几乎不变
            frames.append(ScenarioFrame(cy, ta, cd, ms, tr))
        variants.append(frames)
    scenarios["waving"] = variants

    # ── 6. 蹲下起立 ──
    variants = []
    for v in range(num_variants):
        frames = []
        squat_start = random.randint(20, 25)
        squat_down = random.randint(12, 18)
        squat_up = random.randint(12, 18)
        squat_drop = random.uniform(0.12, 0.18)
        squat_angle = random.uniform(25, 40)

        for t in range(90):
            if t < squat_start:
                cy, ta, cd, ms, tr = _walking_pattern(t)
            elif t < squat_start + squat_down:
                progress = (t - squat_start) / squat_down
                cy = _add_noise(0.35 + progress * squat_drop)
                ta = _add_noise(5.0 + progress * squat_angle)
                cd = _add_noise(0.004 + progress * 0.015)
                ms = _add_noise(0.10 + progress * 0.15)
                tr = _add_noise(0.28 + progress * 0.25)
            elif t < squat_start + squat_down + squat_up:
                progress = (t - squat_start - squat_down) / squat_up
                cy = _add_noise(0.35 + squat_drop * (1 - progress))
                ta = _add_noise(5.0 + squat_angle * (1 - progress))
                cd = _add_noise(0.004)
                ms = _add_noise(0.10)
                tr = _add_noise(0.28)
            else:
                cy, ta, cd, ms, tr = _walking_pattern(t)
            frames.append(ScenarioFrame(cy, ta, cd, ms, tr))
        variants.append(frames)
    scenarios["squatting"] = variants

    return scenarios


# ============================================================
# 评估器
# ============================================================

def evaluate_config(config: FallConfig,
                    scenarios: Dict[str, List[List[ScenarioFrame]]]
                    ) -> Dict[str, float]:
    """
    用给定配置评估所有场景。返回分场景得分 + 总分。
    得分逻辑:
      - 跌倒: 触发且置信度高 = 高分；未触发 = 严重扣分
      - 行走/弯腰/坐下/挥手: 触发 = 扣分，不触发 = 满分
      - 蹲下: 触发 = 小扣分或不扣分（边界场景）

    返回: {"total": float, "walking": ..., "falling": ..., ...}
    """
    scores = {}
    scenario_expectations = {
        "walking":   ("no_alert",  1.0),   # (期望, 权重)
        "falling":   ("alert",     2.0),
        "bending":   ("no_alert",  1.2),
        "sitting":   ("no_alert",  1.0),
        "waving":    ("no_alert",  1.5),
        "squatting": ("no_alert",  0.3),   # 低权重：不强求
    }

    for scenario_name, variants in scenarios.items():
        expectation, weight = scenario_expectations[scenario_name]
        variant_scores = []

        for frames in variants:
            detector = ProcessFallDetector(config=config.process_fall)

            alerts_triggered = []
            for i, f in enumerate(frames):
                features = np.zeros(100, dtype=np.float32)
                features[1] = f.centroid_y
                features[3] = f.torso_angle

                spatial = np.zeros(6, dtype=np.float32)
                spatial[0] = f.centroid_disp
                spatial[3] = f.motion_spread
                spatial[4] = f.torso_disp_ratio

                alert = detector.update(features, spatial, i)
                if alert is not None:
                    alerts_triggered.append(alert)

            # 计算该变体得分
            if expectation == "alert":
                if alerts_triggered:
                    # 有告警：取最高置信度（penalize 低置信）
                    best_conf = max(a.confidence for a in alerts_triggered)
                    best_level = max(a.alert_level for a in alerts_triggered)
                    level_bonus = {"SUSPICIOUS": 0.0, "WARNING": 0.15,
                                   "ALERT": 0.30, "URGENT": 0.50}
                    score = min(1.0, best_conf + level_bonus.get(best_level, 0))
                    # 如果是 SUSPICIOUS 且置信度低，仍然打折
                    if best_level == "SUSPICIOUS" and best_conf < 0.5:
                        score *= 0.6
                else:
                    score = 0.0  # 跌倒未检测到 = 严重失败
            else:
                if alerts_triggered:
                    # 误报：扣分，按置信度和等级
                    worst = min(alerts_triggered,
                                key=lambda a: (0 if a.alert_level == "SUSPICIOUS" else
                                               1 if a.alert_level == "WARNING" else
                                               2 if a.alert_level == "ALERT" else 3))
                    level_penalty = {"SUSPICIOUS": 0.25, "WARNING": 0.55,
                                     "ALERT": 0.80, "URGENT": 1.0}
                    penalty = level_penalty.get(worst.alert_level, 0.5)
                    score = 1.0 - penalty * worst.confidence
                else:
                    score = 1.0  # 正确不告警 = 满分

            variant_scores.append(max(0.0, score))

        scores[scenario_name] = np.mean(variant_scores) * weight

    # 总分 = 加权和
    total_weight = sum(scenario_expectations[s][1] for s in scenarios)
    scores["total"] = sum(scores[s] for s in scenarios) / total_weight
    return scores


# ============================================================
# 参数搜索空间
# ============================================================

PARAM_SEARCH_SPACE = {
    # Motion Spatial (影响 is_likely_fall 静态方法)
    "motion.motion_threshold":            [0.001, 0.002, 0.003, 0.005, 0.008, 0.012],
    "motion.torso_ratio_threshold":       [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
    "motion.spread_alpha":                [0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
    "motion.centroid_disp_weight":        [0.15, 0.25, 0.35, 0.40, 0.50, 0.60],
    "motion.spread_active_weight":        [0.10, 0.20, 0.30, 0.35, 0.45, 0.55],
    "motion.torso_disp_ratio_weight":     [0.10, 0.15, 0.25, 0.35, 0.45, 0.55],

    # Process Fall — 单帧判定
    "process.suspicious_centroid_disp":   [0.01, 0.02, 0.03, 0.04, 0.05, 0.07],
    "process.suspicious_motion_spread":   [0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
    "process.suspicious_torso_disp_ratio":[0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
    "process.suspicious_torso_angle":     [15.0, 20.0, 25.0, 30.0, 35.0, 45.0],
    "process.suspicious_min_signals":     [2, 3, 4],

    # Process Fall — 序列分析
    "process.fall_height_threshold":      [0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
    "process.fall_speed_threshold":       [0.01, 0.02, 0.03, 0.04, 0.05, 0.07],
    "process.min_sequence_frames":        [10, 12, 15, 18, 20, 25],
    "process.consecutive_suspicious":     [2, 3, 4, 5],
    "process.window_seconds":             [1.0, 1.5, 2.0, 2.5, 3.0],

    # Process Fall — 序列分析权重
    "process.analysis_weights_speed":     [0.15, 0.20, 0.25, 0.30, 0.35],
    "process.analysis_weights_height":    [0.20, 0.25, 0.30, 0.35, 0.40],
    "process.analysis_weights_torso":     [0.10, 0.15, 0.20, 0.25, 0.30],

    # Alert thresholds
    "process.alert_threshold_suspicious": [0.25, 0.30, 0.35, 0.40, 0.45],
    "process.alert_threshold_warning":    [0.45, 0.50, 0.55, 0.60, 0.65],
    "process.alert_threshold_alert":      [0.65, 0.70, 0.75, 0.80, 0.85],
    "process.alert_threshold_urgent":     [0.85, 0.90, 0.92, 0.95, 0.97],
}


def _ensure_monotonic(susp: float, warn: float, alrt: float, urg: float) -> Tuple[float, float, float, float]:
    """确保告警阈值单调递增"""
    return (
        min(susp, warn - 0.05),
        max(susp + 0.05, min(warn, alrt - 0.05)),
        max(warn + 0.05, min(alrt, urg - 0.03)),
        max(alrt + 0.03, urg),
    )


def random_params() -> Dict:
    """随机采样一组参数"""
    params = {}
    for key, values in PARAM_SEARCH_SPACE.items():
        params[key] = random.choice(values)
    return params


def apply_params(config: FallConfig, params: Dict) -> FallConfig:
    """将参数字典应用到 config 对象"""
    cfg = deepcopy(config)
    mc = cfg.motion_spatial
    pc = cfg.process_fall

    # Motion Spatial
    if "motion.motion_threshold" in params:
        mc.motion_threshold = params["motion.motion_threshold"]
    if "motion.torso_ratio_threshold" in params:
        mc.torso_ratio_threshold = params["motion.torso_ratio_threshold"]
    if "motion.spread_alpha" in params:
        mc.spread_alpha = params["motion.spread_alpha"]
    if "motion.centroid_disp_weight" in params:
        mc.centroid_disp_weight = params["motion.centroid_disp_weight"]
    if "motion.spread_active_weight" in params:
        mc.spread_active_weight = params["motion.spread_active_weight"]
    if "motion.torso_disp_ratio_weight" in params:
        mc.torso_disp_ratio_weight = params["motion.torso_disp_ratio_weight"]

    # Process Fall — suspicious
    for attr, key in [
        ("suspicious_centroid_disp", "process.suspicious_centroid_disp"),
        ("suspicious_motion_spread", "process.suspicious_motion_spread"),
        ("suspicious_torso_disp_ratio", "process.suspicious_torso_disp_ratio"),
        ("suspicious_torso_angle", "process.suspicious_torso_angle"),
        ("suspicious_min_signals", "process.suspicious_min_signals"),
        ("fall_height_threshold", "process.fall_height_threshold"),
        ("fall_speed_threshold", "process.fall_speed_threshold"),
        ("min_sequence_frames", "process.min_sequence_frames"),
        ("consecutive_suspicious", "process.consecutive_suspicious"),
        ("window_seconds", "process.window_seconds"),
    ]:
        if key in params:
            setattr(pc, attr, params[key])

    # 分析权重
    weights_changed = False
    if "process.analysis_weights_speed" in params:
        pc.analysis_weights["speed"] = params["process.analysis_weights_speed"]
        weights_changed = True
    if "process.analysis_weights_height" in params:
        pc.analysis_weights["height"] = params["process.analysis_weights_height"]
        weights_changed = True
    if "process.analysis_weights_torso" in params:
        pc.analysis_weights["torso"] = params["process.analysis_weights_torso"]
        weights_changed = True
    if weights_changed:
        # 归一化
        total = sum(pc.analysis_weights.values())
        if total > 0:
            for k in pc.analysis_weights:
                pc.analysis_weights[k] /= total

    # 告警阈值（保持单调）
    at = {}
    for k in ("suspicious", "warning", "alert", "urgent"):
        pk = f"process.alert_threshold_{k}"
        if pk in params:
            at[k] = params[pk]
    if at:
        pc.alert_thresholds["SUSPICIOUS"] = at.get("suspicious", pc.alert_thresholds["SUSPICIOUS"])
        pc.alert_thresholds["WARNING"] = at.get("warning", pc.alert_thresholds["WARNING"])
        pc.alert_thresholds["ALERT"] = at.get("alert", pc.alert_thresholds["ALERT"])
        pc.alert_thresholds["URGENT"] = at.get("urgent", pc.alert_thresholds["URGENT"])
        s, w, a, u = _ensure_monotonic(
            pc.alert_thresholds["SUSPICIOUS"],
            pc.alert_thresholds["WARNING"],
            pc.alert_thresholds["ALERT"],
            pc.alert_thresholds["URGENT"],
        )
        pc.alert_thresholds["SUSPICIOUS"] = s
        pc.alert_thresholds["WARNING"] = w
        pc.alert_thresholds["ALERT"] = a
        pc.alert_thresholds["URGENT"] = u

    return cfg


# ============================================================
# 随机搜索
# ============================================================

def random_search(default_config: FallConfig,
                  scenarios: dict,
                  n_iter: int = 4000) -> List[Tuple[float, Dict]]:
    """随机搜索，返回按 total 得分排序的 (score, params) 列表"""
    results = []
    print(f"\n  随机搜索 {n_iter} 组参数...")

    start_time = time.time()
    for i in range(n_iter):
        params = random_params()
        cfg = apply_params(default_config, params)
        scores = evaluate_config(cfg, scenarios)
        results.append((scores["total"], params, scores))

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            best_sofar = max(r[0] for r in results)
            print(f"    [{i+1}/{n_iter}] 耗时 {elapsed:.0f}s | 当前最佳: {best_sofar:.4f}", flush=True)

    results.sort(key=lambda x: x[0], reverse=True)
    return results


# ============================================================
# 精炼 (Refinement)
# ============================================================

def refine_top(default_config: FallConfig,
               scenarios: dict,
               top_candidates: List[Tuple[float, Dict]],
               n_top: int = 20,
               n_perturb: int = 30) -> List[Tuple[float, Dict]]:
    """对 Top-N 候选参数做小范围扰动精炼"""
    print(f"\n  Top-{n_top} 精炼 (每个 {n_perturb} 次扰动)...")

    refined = list(top_candidates[:n_top])  # 保留原始候选
    for rank, (base_score, base_params, _) in enumerate(top_candidates[:n_top]):
        for _ in range(n_perturb):
            perturbed = {}
            for key, value in base_params.items():
                choices = PARAM_SEARCH_SPACE[key]
                # 在相邻值之间随机选
                idx = choices.index(value) if value in choices else len(choices) // 2
                # 小范围扰动
                delta = random.choice([-1, 0, 1])
                new_idx = max(0, min(len(choices) - 1, idx + delta))
                perturbed[key] = choices[new_idx]

            cfg = apply_params(default_config, perturbed)
            scores = evaluate_config(cfg, scenarios)
            refined.append((scores["total"], perturbed, scores))

    refined.sort(key=lambda x: x[0], reverse=True)
    return refined[:50]


# ============================================================
# 报告
# ============================================================

def print_report(results: List[Tuple[float, Dict]], default_config: FallConfig,
                 scenarios: dict, top_n: int = 10):
    """打印 Top-N 结果对比"""
    print(f"\n{'=' * 70}")
    print(f"  TOP {top_n} 参数组合")
    print(f"{'=' * 70}")

    # 默认配置得分
    default_scores = evaluate_config(default_config, scenarios)
    print(f"\n  默认配置 baseline: {default_scores['total']:.4f}")
    print(f"    {'walking':>9}: {default_scores['walking']:5.3f}  "
          f"{'falling':>9}: {default_scores['falling']:5.3f}  "
          f"{'bending':>9}: {default_scores['bending']:5.3f}  "
          f"{'sitting':>9}: {default_scores['sitting']:5.3f}  "
          f"{'waving':>9}: {default_scores['waving']:5.3f}  "
          f"{'squatting':>9}: {default_scores['squatting']:5.3f}")
    print(f"\n  {'─' * 68}")

    for rank, (total, params, scores) in enumerate(results[:top_n]):
        improvement = (total - default_scores["total"]) / max(default_scores["total"], 0.001) * 100
        marker = " *** BEST ***" if rank == 0 else ""
        print(f"\n  #{rank+1}  total={total:.4f}  (+{improvement:+.1f}%){marker}")
        print(f"    {'walking':>9}: {scores['walking']:5.3f}  "
              f"{'falling':>9}: {scores['falling']:5.3f}  "
              f"{'bending':>9}: {scores['bending']:5.3f}  "
              f"{'sitting':>9}: {scores['sitting']:5.3f}  "
              f"{'waving':>9}: {scores['waving']:5.3f}  "
              f"{'squatting':>9}: {scores['squatting']:5.3f}")

        # 关键参数差异
        key_params = [
            "process.fall_height_threshold",
            "process.fall_speed_threshold",
            "process.consecutive_suspicious",
            "process.min_sequence_frames",
            "process.suspicious_min_signals",
        ]
        print(f"    关键参数: ", end="")
        for kp in key_params:
            if kp in params:
                print(f"{kp.split('.')[-1]}={params[kp]}  ", end="")
        print()


def save_best_config(best_params: Dict, default_config: FallConfig,
                     output_path: str, score: float):
    """保存最优配置"""
    best_cfg = apply_params(default_config, best_params)
    best_cfg.description = f"自动调优结果 | 得分={score:.4f}"
    best_cfg.save(output_path)
    print(f"\n  [OK] 最优配置已保存: {output_path}")


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="跌倒识别参数自动调优")
    parser.add_argument("--search", "-n", type=int, default=4000,
                        help="随机搜索迭代次数 (默认 4000)")
    parser.add_argument("--output", "-o", type=str,
                        default=r"E:\老人跌倒\params_best.json",
                        help="最优配置输出路径")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--variants", "-v", type=int, default=15,
                        help="每类场景的变体数量")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式 (1000 搜索, 5 变体)")
    args = parser.parse_args()

    if args.quick:
        args.search = 1000
        args.variants = 5
        print("[快速模式]")

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 70)
    print("  跌倒识别参数自动调优 (Phase C)")
    print("=" * 70)

    # 1. 生成场景
    print(f"\n[1/4] 生成模拟场景 (6类 x {args.variants}变体)...")
    scenarios = generate_scenarios(seed=args.seed, num_variants=args.variants)
    total_frames = sum(sum(len(frames) for frames in variants)
                       for variants in scenarios.values())
    print(f"  场景数: {sum(len(v) for v in scenarios.values())} | "
          f"总帧数: {total_frames}")

    # 2. 默认配置基线
    default_config = FallConfig()
    print(f"\n[2/4] 评估默认配置基线...")
    default_scores = evaluate_config(default_config, scenarios)
    print(f"  baseline total={default_scores['total']:.4f}")

    # 3. 随机搜索
    print(f"\n[3/4] 随机搜索 ({args.search} 次)...")
    search_results = random_search(default_config, scenarios, args.search)

    # 4. 精炼
    print(f"\n[4/4] 精炼 & 输出...")
    refined = refine_top(default_config, scenarios, search_results,
                         n_top=20, n_perturb=30)

    # 报告
    print_report(refined, default_config, scenarios, top_n=10)

    # 保存
    best_score, best_params, best_scores = refined[0]
    save_best_config(best_params, default_config, args.output, best_score)

    # 最终对比
    print(f"\n{'=' * 70}")
    print(f"  调优完成")
    print(f"  baseline:    {default_scores['total']:.4f}")
    print(f"  最优配置:    {best_score:.4f}")
    print(f"  提升幅度:    {(best_score - default_scores['total']) / max(default_scores['total'], 0.001) * 100:+.1f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
