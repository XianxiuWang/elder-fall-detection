#!/usr/bin/env python3
"""
session_analyzer.py — 一键会话分析报告
======================================
用法:
    python session_analyzer.py              # 分析最新 session
    python session_analyzer.py --id 6       # 分析指定 session
    python session_analyzer.py --compare 4 5  # 多 session 对比
    python session_analyzer.py --all         # 所有 session
"""

import sqlite3
import os
import sys
import json
import argparse
from collections import Counter
from datetime import datetime
import numpy as np

DB_PATH = r"E:\老人跌倒\logs\activity.db"


class SessionAnalyzer:
    def __init__(self, db_path=DB_PATH):
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row

    def get_session(self, session_id: int):
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def get_all_sessions(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM sessions ORDER BY id").fetchall()]

    def get_frame_logs(self, session_id: int):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM frame_logs WHERE session_id=? ORDER BY id", (session_id,)).fetchall()]

    def get_alerts(self, session_id: int):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM alerts WHERE session_id=? ORDER BY id", (session_id,)).fetchall()]

    def analyze(self, session_id: int):
        sess = self.get_session(session_id)
        if not sess:
            return f"[ERROR] Session #{session_id} not found"

        frames = self.get_frame_logs(session_id)
        alerts = self.get_alerts(session_id)

        # 兼容会话被中断导致 total_frames=0 的情况
        total_frames = sess['total_frames']
        if total_frames <= 0 and frames:
            total_frames = frames[-1]['frame_idx']
        fps = sess['fps']
        dur_sec = total_frames / fps if fps > 0 else 0
        proc_fps = len(frames) / dur_sec if dur_sec > 0 else 0

        lines = []
        lines.append(f"{'='*60}")
        lines.append(f"  Session #{session_id} Analysis Report")
        lines.append(f"{'='*60}")
        lines.append(f"  Started:    {sess['start_time']}")
        lines.append(f"  Ended:      {sess.get('end_time', 'N/A')}")
        lines.append(f"  Duration:   {dur_sec:.1f}s ({dur_sec/60:.1f}min)")
        lines.append(f"  Frames:     {total_frames} total | {len(frames)} logged "
                     f"({len(frames)/max(total_frames,1)*100:.0f}%)")
        lines.append(f"  FPS:        {sess['fps']} (logged ~{proc_fps:.1f}/s)")
        lines.append(f"  Alerts:     {len(alerts)}")

        if not frames:
            lines.append("\n  (no frame logs)")
            return "\n".join(lines)

        # Risk score
        scores = [f['risk_score'] for f in frames if f['risk_score'] is not None]
        if scores:
            arr = np.array(scores)
            lines.append(f"\n  ── Risk Score ──")
            lines.append(f"    Min={arr.min():.1f}  Max={arr.max():.1f}")
            lines.append(f"    Mean={arr.mean():.1f}  Median={np.median(arr):.1f}  Std={arr.std():.1f}")

            # Drift
            if len(arr) > 10:
                q = max(1, len(arr)//10)
                first, last = np.mean(arr[:q]), np.mean(arr[-q:])
                drift = last - first
                sym = "↑ 上升" if drift > 3 else ("↓ 下降" if drift < -3 else "→ 平稳")
                lines.append(f"    Start→End: {first:.1f}→{last:.1f}  ({drift:+.1f})  {sym}")

                # Quartile trajectory
                q25, q50, q75 = np.percentile(arr, [25, 50, 75])
                lines.append(f"    Quartiles: Q25={q25:.1f}  Q50={q50:.1f}  Q75={q75:.1f}")

        # Risk levels
        levels = Counter(f['alert_level'] for f in frames if f['alert_level'])
        if levels:
            total = sum(levels.values())
            lines.append(f"\n  ── Risk Levels ──")
            for lvl in ['SAFE', 'BLUE', 'YELLOW', 'ORANGE', 'RED']:
                cnt = levels.get(lvl, 0)
                pct = cnt / total * 100
                bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
                lines.append(f"    {lvl:8s} {cnt:4d} ({pct:5.1f}%) {bar}")

        # ML predictions
        ml_classes = Counter(f['ml_class_name'] for f in frames if f['ml_class_name'])
        if ml_classes:
            total_ml = sum(ml_classes.values())
            coverage = total_ml / len(frames) * 100
            lines.append(f"\n  ── ML Predictions ({total_ml} predictions, {coverage:.1f}% coverage) ──")
            for cls, cnt in ml_classes.most_common():
                lines.append(f"    {cls:10s} {cnt:4d} ({cnt/total_ml*100:5.1f}%)")

        # Risk sub-components
        sub_risks = {}
        for key in ['sway_risk', 'com_bos_risk', 'gait_risk', 'posture_risk', 'transition_risk']:
            vals = [f[key] for f in frames if f[key] is not None]
            if vals:
                sub_risks[key] = {'mean': np.mean(vals), 'max': np.max(vals)}
        if sub_risks:
            lines.append(f"\n  ── Risk Sub-Components ──")
            for k, v in sub_risks.items():
                label = k.replace('_risk', '').replace('_', ' ').title()
                lines.append(f"    {label:15s} mean={v['mean']:5.1f}  max={v['max']:5.1f}")

        # Biomechanical
        sway_vals = [f['trunk_sway_deg'] for f in frames if f['trunk_sway_deg']]
        if sway_vals:
            lines.append(f"\n  ── Biomechanics ──")
            lines.append(f"    Trunk Sway:  mean={np.mean(sway_vals):.1f}°  max={np.max(sway_vals):.1f}°")

        # Alerts
        if alerts:
            alert_levels = Counter(a['alert_level'] for a in alerts)
            alert_types = Counter(a['alert_type'] for a in alerts)
            lines.append(f"\n  ── Alert Breakdown ({len(alerts)} total) ──")
            for lvl, cnt in sorted(alert_levels.items()):
                lines.append(f"    {lvl:8s} {cnt:4d}")
            lines.append(f"    Types: {dict(alert_types)}")

        lines.append(f"\n{'='*60}")
        return "\n".join(lines)

    def compare(self, session_ids: list):
        """对比多个 session"""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"  Multi-Session Comparison: {session_ids}")
        lines.append(f"{'='*60}")

        # Header
        header = f"  {'Session':>8s} {'Duration':>8s} {'Frames':>8s} {'Risk均值':>8s} {'Risk漂移':>8s} {'告警':>6s} {'SAFE%':>7s} {'YELL+%':>7s}"
        lines.append(header)
        lines.append(f"  {'-'*65}")

        for sid in session_ids:
            sess = self.get_session(sid)
            if not sess:
                continue
            frames = self.get_frame_logs(sid)
            alerts = self.get_alerts(sid)
            dur = total_frames / sess['fps'] if sess['fps'] else 0
            frames = self.get_frame_logs(sid)
            alerts = self.get_alerts(sid)
            total_frames = sess['total_frames']
            if total_frames <= 0 and frames:
                total_frames = frames[-1]['frame_idx']
            dur = total_frames / sess['fps'] if sess['fps'] else 0

            scores = [f['risk_score'] for f in frames if f['risk_score'] is not None]
            levels = Counter(f['alert_level'] for f in frames if f['alert_level'])
            total_lvl = sum(levels.values())

            if scores:
                mean_s = f"{np.mean(scores):.1f}"
                drift = ""
                if len(scores) > 10:
                    q = max(1, len(scores)//10)
                    d = np.mean(scores[-q:]) - np.mean(scores[:q])
                    drift = f"{d:+.1f}"
                safe_pct = levels.get('SAFE', 0) / total_lvl * 100 if total_lvl else 0
                yell_pct = (levels.get('YELLOW', 0) + levels.get('ORANGE', 0) + levels.get('RED', 0)) / total_lvl * 100 if total_lvl else 0
                lines.append(
                    f"  #{sid:>6d} {dur:>3.0f}s{'' if dur<60 else f'({dur/60:.1f}min)':<4s} {sess['total_frames']:>8d} {mean_s:>8s} {drift:>8s} {len(alerts):>6d} {safe_pct:>6.1f}% {yell_pct:>6.1f}%"
                )
            else:
                lines.append(f"  #{sid:>6d} (no data)")

        lines.append(f"{'='*60}\n")

        # Trend note
        if len(session_ids) >= 3:
            last3 = session_ids[-3:]
            risk_means = []
            alert_counts = []
            for sid in last3:
                frames = self.get_frame_logs(sid)
                scores = [f['risk_score'] for f in frames if f['risk_score'] is not None]
                alerts = self.get_alerts(sid)
                risk_means.append(np.mean(scores) if scores else 0)
                alert_counts.append(len(alerts))
            lines.append(f"  Recent Trend (last {len(last3)} sessions):")
            lines.append(f"    Risk means: {[f'{x:.1f}' for x in risk_means]}")
            lines.append(f"    Alert counts: {alert_counts}")
            if len(risk_means) >= 3:
                trend = "↓ Improving" if risk_means[-1] < risk_means[0] else "↑ Degrading"
                lines.append(f"    Overall: {trend}")

        return "\n".join(lines)

    def close(self):
        self.db.close()


def main():
    parser = argparse.ArgumentParser(description="Session Analyzer")
    parser.add_argument("--id", type=int, help="Analyze specific session")
    parser.add_argument("--compare", type=int, nargs="+", help="Compare sessions")
    parser.add_argument("--all", action="store_true", help="Analyze all sessions")
    parser.add_argument("--latest", action="store_true", help="Analyze latest session")
    args = parser.parse_args()

    if not any([args.id, args.compare, args.all, args.latest]):
        args.latest = True

    a = SessionAnalyzer()

    if args.compare:
        print(a.compare(args.compare))
    elif args.all:
        sessions = a.get_all_sessions()
        if sessions:
            print(a.compare([s['id'] for s in sessions]))
            print()
            for s in sessions:
                print(a.analyze(s['id']))
                print()
        else:
            print("No sessions found")
    elif args.id:
        print(a.analyze(args.id))
    elif args.latest:
        sessions = a.get_all_sessions()
        if sessions:
            print(a.analyze(sessions[-1]['id']))
        else:
            print("No sessions found")

    a.close()


if __name__ == "__main__":
    main()
