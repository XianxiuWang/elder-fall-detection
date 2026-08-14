#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
activity_logger.py — 活动日志数据库 v1.0
========================================
SQLite 记录每一帧的 risk_score、六维风险、分类结果、时间戳。

功能:
  1. 实时帧级日志 (批量写入, 低开销)
  2. 查询"过去 24 小时最高风险片段"
  3. 导出异常片段为 MP4
  4. 长期趋势分析 (日/周/月报表)

用法:
    logger = ActivityLogger("E:/老人跌倒/logs")
    logger.start_session()
    # 每帧调用
    logger.log_frame(risk_report, ml_result, frame_idx, timestamp)
    # 结束时
    logger.end_session()

    # 查询
    segments = logger.query_high_risk_segments(hours=24, min_risk=50)
    logger.export_anomaly_video(segments[0], source_video_path, output_path)

数据库位置: E:\老人跌倒\logs\activity.db

设计原则:
  - WAL 模式 + 批量写入, 不对实时管线产生可感知延迟
  - 默认每 30 帧写一次, 避免每帧 fsync
  - 查询和分析走独立连接, 不阻塞写入
"""

import sqlite3
import os
import time
import json
import struct
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

# ── 兼容: 如果不在完整环境中也能运行分析脚本 ──
try:
    import numpy as np
except ImportError:
    np = None

# ── 数据类 ──

@dataclass
class FrameRecord:
    """一条帧日志 (结构化, 方便传递)"""
    session_id: int = 0
    frame_idx: int = 0
    timestamp: float = 0.0
    iso_time: str = ""

    # 综合风险
    risk_score: float = 0.0
    alert_level: str = "SAFE"

    # 六维风险
    sway_risk: float = 0.0
    com_bos_risk: float = 0.0
    gait_risk: float = 0.0
    posture_risk: float = 0.0
    transition_risk: float = 0.0
    direction_boost: float = 0.0

    # 关键生物力学
    trunk_sway_deg: float = 0.0
    trunk_tilt_deg: float = 0.0
    com_bos_margin: float = 1.0
    bbox_ratio: float = 1.0
    angular_velocity_max: float = 0.0

    # 跌倒方向
    fall_direction: str = "none"

    # ML 分类
    ml_class_id: int = -1
    ml_class_name: str = ""
    ml_fall_prob: float = 0.0
    ml_fall_triggered: bool = False

    # 告警
    alert_active: bool = False


@dataclass
class RiskSegment:
    """一段高风险片段"""
    segment_id: int
    session_id: int
    session_start: str
    start_time: str
    end_time: str
    start_frame: int
    end_frame: int
    duration_sec: float
    peak_risk: float
    avg_risk: float
    alert_level: str
    ml_fall_triggered: bool
    fall_direction: str


@dataclass
class DailySummary:
    """日度摘要"""
    date: str
    total_frames: int
    total_duration_sec: float
    avg_risk: float
    max_risk: float
    alert_count_blue: int
    alert_count_yellow: int
    alert_count_orange: int
    alert_count_red: int
    ml_fall_count: int
    dominant_activity: str


# ── 主类 ──

class ActivityLogger:
    """活动日志数据库管理器"""

    # 批量写入配置
    FLUSH_INTERVAL_FRAMES = 30   # 每 30 帧写一次
    FLUSH_INTERVAL_SECONDS = 2   # 或每 2 秒写一次 (取先到者)
    LOG_EVERY_N_FRAMES = 3       # 每 3 帧记录一次 (可调)

    def __init__(self, log_dir: str = r"E:\老人跌倒\logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.db_path = os.path.join(log_dir, "activity.db")
        self.video_dir = os.path.join(log_dir, "anomaly_clips")
        os.makedirs(self.video_dir, exist_ok=True)

        self._init_db()
        self._write_buf: List[Tuple] = []
        self._last_flush_time = time.time()
        self._session_id: Optional[int] = None
        self._session_start: Optional[float] = None
        self._frame_counter: int = 0

    # ════════════════════════════════════════════════════
    # 数据库初始化
    # ════════════════════════════════════════════════════

    def _init_db(self):
        """创建表结构 (幂等: IF NOT EXISTS)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")       # 写入不锁读
            conn.execute("PRAGMA synchronous=NORMAL")      # 性能优先
            conn.execute("PRAGMA cache_size=-8000")        # 8MB 缓存
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time    TEXT NOT NULL,
                    end_time      TEXT,
                    duration_sec  REAL,
                    total_frames  INTEGER DEFAULT 0,
                    source        TEXT DEFAULT 'camera',
                    fps           REAL DEFAULT 15.0
                );

                CREATE TABLE IF NOT EXISTS frame_logs (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id    INTEGER NOT NULL,
                    frame_idx     INTEGER NOT NULL,
                    timestamp     REAL NOT NULL,
                    iso_time      TEXT NOT NULL,

                    -- 综合风险
                    risk_score    REAL DEFAULT 0.0,
                    alert_level   TEXT DEFAULT 'SAFE',

                    -- 六维风险
                    sway_risk     REAL DEFAULT 0.0,
                    com_bos_risk  REAL DEFAULT 0.0,
                    gait_risk     REAL DEFAULT 0.0,
                    posture_risk  REAL DEFAULT 0.0,
                    transition_risk REAL DEFAULT 0.0,
                    direction_boost REAL DEFAULT 0.0,

                    -- 生物力学
                    trunk_sway_deg   REAL DEFAULT 0.0,
                    trunk_tilt_deg   REAL DEFAULT 0.0,
                    com_bos_margin   REAL DEFAULT 1.0,
                    bbox_ratio       REAL DEFAULT 1.0,
                    angular_vel_max  REAL DEFAULT 0.0,
                    fall_direction   TEXT DEFAULT 'none',

                    -- ML分类
                    ml_class_id      INTEGER DEFAULT -1,
                    ml_class_name    TEXT DEFAULT '',
                    ml_fall_prob     REAL DEFAULT 0.0,
                    ml_fall_triggered INTEGER DEFAULT 0,

                    -- 告警状态
                    alert_active     INTEGER DEFAULT 0,

                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_frame_session
                    ON frame_logs(session_id, frame_idx);
                CREATE INDEX IF NOT EXISTS idx_frame_risk
                    ON frame_logs(risk_score DESC);
                CREATE INDEX IF NOT EXISTS idx_frame_time
                    ON frame_logs(iso_time);
                CREATE INDEX IF NOT EXISTS idx_frame_alert
                    ON frame_logs(alert_level, iso_time);

                CREATE TABLE IF NOT EXISTS alerts (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id    INTEGER NOT NULL,
                    frame_idx     INTEGER NOT NULL,
                    iso_time      TEXT NOT NULL,
                    alert_type    TEXT NOT NULL,  -- 'risk_orange', 'risk_red', 'ml_fall'
                    risk_score    REAL,
                    alert_level   TEXT,
                    ml_class_name TEXT,
                    fall_direction TEXT,
                    details       TEXT,           -- JSON extra info
                    acknowledged  INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE INDEX IF NOT EXISTS idx_alert_time
                    ON alerts(iso_time);

                CREATE TABLE IF NOT EXISTS daily_summaries (
                    date          TEXT PRIMARY KEY,
                    total_frames  INTEGER DEFAULT 0,
                    total_duration_sec REAL DEFAULT 0,
                    avg_risk      REAL DEFAULT 0.0,
                    max_risk      REAL DEFAULT 0.0,
                    alert_blue    INTEGER DEFAULT 0,
                    alert_yellow  INTEGER DEFAULT 0,
                    alert_orange  INTEGER DEFAULT 0,
                    alert_red     INTEGER DEFAULT 0,
                    ml_fall_count INTEGER DEFAULT 0,
                    dominant_activity TEXT DEFAULT '',
                    summary_json  TEXT DEFAULT '{}'
                );
            """)

    # ════════════════════════════════════════════════════
    # 会话管理
    # ════════════════════════════════════════════════════

    def start_session(self, source: str = "camera", fps: float = 15.0) -> int:
        """开始新监控会话, 返回 session_id"""
        now = datetime.now().isoformat(timespec='milliseconds')
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO sessions (start_time, source, fps) VALUES (?, ?, ?)",
                (now, source, fps)
            )
            self._session_id = cur.lastrowid
        self._session_start = time.time()
        self._frame_counter = 0
        self._write_buf.clear()
        self._last_flush_time = time.time()
        print(f"  [LOG] Session #{self._session_id} started @ {now}")
        return self._session_id

    def end_session(self):
        """结束当前会话, 刷写剩余缓冲"""
        self._flush()
        if self._session_id is not None:
            now = datetime.now().isoformat(timespec='milliseconds')
            duration = time.time() - self._session_start if self._session_start else 0
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE sessions SET end_time=?, duration_sec=?, total_frames=? WHERE id=?",
                    (now, duration, self._frame_counter, self._session_id)
                )
                # 自动生成日度摘要
                self._generate_daily_summary(conn, self._session_id)
            print(f"  [LOG] Session #{self._session_id} ended. "
                  f"Duration: {duration:.0f}s, Frames: {self._frame_counter}")
        self._session_id = None

    # ════════════════════════════════════════════════════
    # 帧日志 (批量缓冲)
    # ════════════════════════════════════════════════════

    def log_frame(self, risk_report=None, ml_result=None,
                  frame_idx: int = 0, timestamp: float = 0.0,
                  alert_active: bool = False):
        """
        记录一帧数据。内部做下采样 + 批量写入。

        Args:
            risk_report: FallRiskReport 对象 (来自 fall_predictor)
            ml_result:   DetectionResult 对象 (来自 ml_5class_detector)
            frame_idx:   帧序号
            timestamp:   Unix 时间戳
            alert_active: 当前是否有活跃告警
        """
        if self._session_id is None:
            return

        self._frame_counter = frame_idx

        # 下采样: 每 N 帧记录一次
        if frame_idx % self.LOG_EVERY_N_FRAMES != 0:
            return

        iso_time = datetime.now().isoformat(timespec='milliseconds')
        if timestamp <= 0:
            timestamp = time.time()

        # ── 从 risk_report 提取数据 ──
        if risk_report is not None:
            r = risk_report
            risk_score = r.risk_score
            alert_level = r.alert_level
            sway_risk = r.sway_risk
            com_bos_risk = r.com_bos_risk
            gait_risk = r.gait_risk
            posture_risk = r.posture_risk
            transition_risk = r.transition_risk
            direction_boost = r.direction_risk_boost
            trunk_sway_deg = r.trunk_sway_deg
            trunk_tilt_deg = r.trunk_tilt_deg
            com_bos_margin = r.com_bos_margin
            bbox_ratio = r.bbox_ratio
            angular_vel_max = r.angular_velocity_max
            fall_direction = r.fall_direction
        else:
            risk_score = alert_level = sway_risk = com_bos_risk = gait_risk = \
                posture_risk = transition_risk = direction_boost = trunk_sway_deg = \
                trunk_tilt_deg = com_bos_margin = bbox_ratio = angular_vel_max = 0.0
            alert_level = "SAFE"
            fall_direction = "none"

        # ── 从 ml_result 提取数据 ──
        if ml_result is not None and ml_result.inference_done:
            ml_class_id = ml_result.class_id
            ml_class_name = ml_result.class_name
            ml_fall_prob = ml_result.fall_prob
            ml_fall_triggered = 1 if ml_result.fall_triggered else 0
        else:
            ml_class_id = -1
            ml_class_name = ""
            ml_fall_prob = 0.0
            ml_fall_triggered = 0

        alert_flag = 1 if alert_active else 0

        # ── 加入缓冲 ──
        self._write_buf.append((
            self._session_id, frame_idx, timestamp, iso_time,
            risk_score, alert_level,
            sway_risk, com_bos_risk, gait_risk, posture_risk,
            transition_risk, direction_boost,
            trunk_sway_deg, trunk_tilt_deg, com_bos_margin,
            bbox_ratio, angular_vel_max, fall_direction,
            ml_class_id, ml_class_name, ml_fall_prob, ml_fall_triggered,
            alert_flag
        ))

        # ── 触发告警记录 ──
        if alert_level in ("ORANGE", "RED") or ml_fall_triggered:
            self._log_alert(frame_idx, iso_time, risk_score,
                            alert_level, ml_class_name,
                            fall_direction,
                            ml_fall_triggered)

        # ── 定时刷写 ──
        if (len(self._write_buf) >= self.FLUSH_INTERVAL_FRAMES or
            time.time() - self._last_flush_time >= self.FLUSH_INTERVAL_SECONDS):
            self._flush()

    def _flush(self):
        """将缓冲批量写入数据库"""
        if not self._write_buf:
            return

        sql = """INSERT INTO frame_logs (
            session_id, frame_idx, timestamp, iso_time,
            risk_score, alert_level,
            sway_risk, com_bos_risk, gait_risk, posture_risk,
            transition_risk, direction_boost,
            trunk_sway_deg, trunk_tilt_deg, com_bos_margin,
            bbox_ratio, angular_vel_max, fall_direction,
            ml_class_id, ml_class_name, ml_fall_prob, ml_fall_triggered,
            alert_active
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany(sql, self._write_buf)
            self._write_buf.clear()
            self._last_flush_time = time.time()
        except Exception as e:
            print(f"  [LOG ERROR] 批量写入失败: {e}")

    def _log_alert(self, frame_idx: int, iso_time: str,
                   risk_score: float, alert_level: str,
                   ml_class_name: str, fall_direction: str,
                   ml_fall_triggered: int):
        """记录告警事件"""
        if ml_fall_triggered:
            alert_type = "ml_fall"
        elif alert_level == "RED":
            alert_type = "risk_red"
        elif alert_level == "ORANGE":
            alert_type = "risk_orange"
        else:
            return  # 不记录低级别

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO alerts (session_id, frame_idx, iso_time,
                       alert_type, risk_score, alert_level, ml_class_name,
                       fall_direction, details)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (self._session_id, frame_idx, iso_time,
                     alert_type, risk_score, alert_level,
                     ml_class_name, fall_direction, "{}")
                )
        except Exception as e:
            print(f"  [LOG ERROR] 告警写入失败: {e}")

    # ════════════════════════════════════════════════════
    # 查询: 高风险片段
    # ════════════════════════════════════════════════════

    def query_high_risk_segments(self, hours: int = 24, min_risk: float = 40.0,
                                 top_n: int = 10) -> List[RiskSegment]:
        """
        查询过去 N 小时内风险最高的片段。

        Args:
            hours: 时间范围 (小时)
            min_risk: 最低风险阈值
            top_n: 返回前 N 个片段
        Returns:
            List[RiskSegment], 按 peak_risk 降序
        """
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            # 找出所有 risk_score >= min_risk 的帧, 按时间分组为连续片段
            rows = conn.execute("""
                SELECT f.session_id, f.frame_idx, f.iso_time, f.risk_score,
                       f.alert_level, f.ml_fall_triggered, f.fall_direction,
                       s.start_time as session_start
                FROM frame_logs f
                JOIN sessions s ON f.session_id = s.id
                WHERE f.iso_time >= ? AND f.risk_score >= ?
                ORDER BY f.iso_time ASC
            """, (cutoff, min_risk)).fetchall()

        if not rows:
            return []

        # 将连续帧合并为片段
        segments = []
        current_seg = None
        seg_id = 0

        for row in rows:
            fid = row['frame_idx']
            if current_seg is None:
                current_seg = self._new_segment(seg_id, row)
            elif fid - current_seg.end_frame > 15:  # 间隔 >15 帧 = 新片段
                segments.append(current_seg)
                seg_id += 1
                current_seg = self._new_segment(seg_id, row)
            else:
                # 扩展当前片段
                current_seg.end_frame = fid
                current_seg.end_time = row['iso_time']
                current_seg.duration_sec = self._time_diff(
                    current_seg.start_time, current_seg.end_time
                )
                if row['risk_score'] > current_seg.peak_risk:
                    current_seg.peak_risk = row['risk_score']
                    current_seg.alert_level = row['alert_level']
                current_seg.avg_risk = (current_seg.avg_risk + row['risk_score']) / 2
                if row['ml_fall_triggered']:
                    current_seg.ml_fall_triggered = True
                if row['fall_direction'] != 'none':
                    current_seg.fall_direction = row['fall_direction']

        if current_seg is not None:
            segments.append(current_seg)

        # 排序 + Top-N
        segments.sort(key=lambda s: s.peak_risk, reverse=True)
        return segments[:top_n]

    def _new_segment(self, seg_id: int, row) -> RiskSegment:
        return RiskSegment(
            segment_id=seg_id,
            session_id=row['session_id'],
            session_start=row['session_start'],
            start_time=row['iso_time'],
            end_time=row['iso_time'],
            start_frame=row['frame_idx'],
            end_frame=row['frame_idx'],
            duration_sec=0.0,
            peak_risk=row['risk_score'],
            avg_risk=row['risk_score'],
            alert_level=row['alert_level'],
            ml_fall_triggered=bool(row['ml_fall_triggered']),
            fall_direction=row['fall_direction'],
        )

    @staticmethod
    def _time_diff(t1: str, t2: str) -> float:
        """计算两个 ISO 时间字符串的时间差 (秒)"""
        try:
            dt1 = datetime.fromisoformat(t1)
            dt2 = datetime.fromisoformat(t2)
            return (dt2 - dt1).total_seconds()
        except:
            return 0.0

    # ════════════════════════════════════════════════════
    # 查询: 最近的告警
    # ════════════════════════════════════════════════════

    def query_recent_alerts(self, hours: int = 24, limit: int = 50) -> List[Dict]:
        """查询最近的告警记录"""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM alerts
                WHERE iso_time >= ?
                ORDER BY iso_time DESC
                LIMIT ?
            """, (cutoff, limit)).fetchall()
        return [dict(r) for r in rows]

    # ════════════════════════════════════════════════════
    # 导出: 异常片段 → MP4
    # ════════════════════════════════════════════════════

    def export_anomaly_video(self, segment: RiskSegment,
                             source_video_path: str,
                             output_path: Optional[str] = None,
                             context_sec: float = 3.0) -> Optional[str]:
        """
        导出一个高风险片段为 MP4 视频文件。

        Args:
            segment:   RiskSegment 对象
            source_video_path: 原始监控视频路径 (或 "camera")
            output_path: 输出路径, None 则自动生成
            context_sec: 片段前后扩展上下文秒数
        Returns:
            输出文件路径, 或 None (如果 source 是 camera)
        """
        if source_video_path == "camera":
            print("  [LOG] 摄像头源无法导出历史片段")
            return None

        if not os.path.exists(source_video_path):
            print(f"  [LOG] 源视频不存在: {source_video_path}")
            return None

        if output_path is None:
            ts = segment.start_time.replace(':', '-').replace('.', '-')
            output_path = os.path.join(
                self.video_dir,
                f"anomaly_s{segment.session_id}_f{segment.start_frame}-{segment.end_frame}_{ts}.mp4"
            )

        try:
            import subprocess
            # 估算 FPS (从帧数 + 时长推算)
            if segment.duration_sec > 0:
                fps = segment.end_frame / segment.duration_sec
            else:
                fps = 15.0

            # 计算起止时间
            start_sec = max(0, segment.start_frame / fps - context_sec)
            duration = (segment.end_frame - segment.start_frame) / fps + context_sec * 2

            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", str(start_sec),
                "-i", source_video_path,
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            subprocess.run(cmd, check=True, timeout=30)
            print(f"  [LOG] 异常片段已导出: {output_path}")
            return output_path
        except Exception as e:
            print(f"  [LOG] 导出失败: {e}")
            return None

    # ════════════════════════════════════════════════════
    # 趋势分析
    # ════════════════════════════════════════════════════

    def get_daily_summaries(self, days: int = 7) -> List[DailySummary]:
        """获取最近 N 天的日度摘要"""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM daily_summaries
                WHERE date >= ?
                ORDER BY date DESC
            """, (cutoff,)).fetchall()

        if rows:
            return [DailySummary(
                date=r['date'],
                total_frames=r['total_frames'],
                total_duration_sec=r['total_duration_sec'],
                avg_risk=r['avg_risk'],
                max_risk=r['max_risk'],
                alert_count_blue=r['alert_blue'],
                alert_count_yellow=r['alert_yellow'],
                alert_count_orange=r['alert_orange'],
                alert_count_red=r['alert_red'],
                ml_fall_count=r['ml_fall_count'],
                dominant_activity=r['dominant_activity'],
            ) for r in rows]
        return []

    def get_trend_data(self, days: int = 7) -> Dict[str, Any]:
        """
        获取长期趋势数据, 用于前端图表展示。

        Returns:
            {
                "dates": ["2026-07-20", ...],
                "avg_risk": [12.3, 15.1, ...],
                "max_risk": [45.0, 52.0, ...],
                "alert_counts": {"blue": [...], "yellow": [...], "orange": [...], "red": [...]},
                "ml_fall_counts": [0, 0, 1, ...],
                "trend": "stable" | "improving" | "worsening" | "insufficient_data"
            }
        """
        summaries = self.get_daily_summaries(days)

        if len(summaries) < 2:
            return {"trend": "insufficient_data", "message": "需要至少 2 天数据"}

        # 按日期正序
        summaries.sort(key=lambda s: s.date)

        dates = [s.date for s in summaries]
        avg_risks = [s.avg_risk for s in summaries]
        max_risks = [s.max_risk for s in summaries]
        alert_counts = {
            "blue": [s.alert_count_blue for s in summaries],
            "yellow": [s.alert_count_yellow for s in summaries],
            "orange": [s.alert_count_orange for s in summaries],
            "red": [s.alert_count_red for s in summaries],
        }
        ml_falls = [s.ml_fall_count for s in summaries]

        # 简单趋势判断: 线性回归斜率
        if len(avg_risks) >= 3:
            x = np.array(range(len(avg_risks)))
            y = np.array(avg_risks)
            slope = np.polyfit(x, y, 1)[0]
            if slope > 1.0:
                trend = "worsening"
            elif slope < -1.0:
                trend = "improving"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "dates": dates,
            "avg_risk": avg_risks,
            "max_risk": max_risks,
            "alert_counts": alert_counts,
            "ml_fall_counts": ml_falls,
            "trend": trend,
        }

    def get_risk_deterioration_alert(self, days: int = 7, warning_threshold: float = 3.0) -> Optional[str]:
        """
        检测老人的跌倒风险是否在恶化。

        如果 avg_risk 的 7 日线性回归斜率 > warning_threshold 且最近 2 天连续上升,
        返回告警消息。否则返回 None。
        """
        trend = self.get_trend_data(days)
        if trend["trend"] != "worsening":
            return None

        # 检查最近 2 天是否连续上升
        avg_risks = trend["avg_risk"]
        if len(avg_risks) < 3:
            return None
        if avg_risks[-1] > avg_risks[-2] > avg_risks[-3]:
            return (f"⚠️ 老人跌倒风险持续上升: 近 3 天日均风险 "
                    f"{avg_risks[-3]:.1f} → {avg_risks[-2]:.1f} → {avg_risks[-1]:.1f}, "
                    f"建议关注")
        return None

    # ════════════════════════════════════════════════════
    # 日度摘要生成
    # ════════════════════════════════════════════════════

    def _generate_daily_summary(self, conn, session_id: int):
        """为当前会话生成日度摘要 (内部调用)"""
        today = datetime.now().strftime("%Y-%m-%d")

        row = conn.execute("""
            SELECT
                COUNT(*) as total_frames,
                AVG(risk_score) as avg_risk,
                MAX(risk_score) as max_risk,
                SUM(CASE WHEN alert_level = 'BLUE' THEN 1 ELSE 0 END) as blue_cnt,
                SUM(CASE WHEN alert_level = 'YELLOW' THEN 1 ELSE 0 END) as yellow_cnt,
                SUM(CASE WHEN alert_level = 'ORANGE' THEN 1 ELSE 0 END) as orange_cnt,
                SUM(CASE WHEN alert_level = 'RED' THEN 1 ELSE 0 END) as red_cnt,
                SUM(ml_fall_triggered) as fall_cnt,
                MAX(iso_time) as max_time,
                MIN(iso_time) as min_time
            FROM frame_logs
            WHERE session_id = ? AND iso_time LIKE ?
        """, (session_id, f"{today}%")).fetchone()

        if row is None or row[0] == 0:
            return

        total_frames = row[0]
        avg_risk = row[1] or 0
        max_risk = row[2] or 0
        blue = row[3] or 0
        yellow = row[4] or 0
        orange = row[5] or 0
        red = row[6] or 0
        fall_cnt = row[7] or 0

        # 计算时长
        try:
            t_min = datetime.fromisoformat(row[9]) if row[9] else datetime.now()
            t_max = datetime.fromisoformat(row[8]) if row[8] else datetime.now()
            duration = (t_max - t_min).total_seconds()
        except:
            duration = 0

        # 主导活动
        dom = conn.execute("""
            SELECT ml_class_name, COUNT(*) as cnt
            FROM frame_logs
            WHERE session_id = ? AND ml_class_name != ''
            GROUP BY ml_class_name
            ORDER BY cnt DESC LIMIT 1
        """, (session_id,)).fetchone()
        dominant = dom[0] if dom else ""

        conn.execute("""
            INSERT INTO daily_summaries
                (date, total_frames, total_duration_sec, avg_risk, max_risk,
                 alert_blue, alert_yellow, alert_orange, alert_red,
                 ml_fall_count, dominant_activity)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(date) DO UPDATE SET
                total_frames = total_frames + excluded.total_frames,
                total_duration_sec = total_duration_sec + excluded.total_duration_sec,
                avg_risk = (avg_risk + excluded.avg_risk) / 2,
                max_risk = MAX(max_risk, excluded.max_risk),
                alert_blue = alert_blue + excluded.alert_blue,
                alert_yellow = alert_yellow + excluded.alert_yellow,
                alert_orange = alert_orange + excluded.alert_orange,
                alert_red = alert_red + excluded.alert_red,
                ml_fall_count = ml_fall_count + excluded.ml_fall_count,
                dominant_activity = CASE WHEN excluded.dominant_activity != ''
                    THEN excluded.dominant_activity ELSE dominant_activity END
        """, (today, total_frames, duration, avg_risk, max_risk,
              blue, yellow, orange, red, fall_cnt, dominant))

    # ════════════════════════════════════════════════════
    # 统计查询
    # ════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """获取总体统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            frames = conn.execute("SELECT COUNT(*) as cnt FROM frame_logs").fetchone()
            sessions = conn.execute("SELECT COUNT(*) as cnt FROM sessions").fetchone()
            alerts = conn.execute(
                "SELECT COUNT(*) as cnt FROM alerts WHERE iso_time >= ?",
                ((datetime.now() - timedelta(hours=24)).isoformat(),)
            ).fetchone()
            last = conn.execute(
                "SELECT iso_time, risk_score, alert_level FROM frame_logs ORDER BY id DESC LIMIT 1"
            ).fetchone()

        return {
            "total_frames": frames['cnt'],
            "total_sessions": sessions['cnt'],
            "alerts_24h": alerts['cnt'],
            "last_log": {
                "time": last['iso_time'] if last else None,
                "risk": last['risk_score'] if last else None,
                "alert": last['alert_level'] if last else None,
            } if last else None,
            "db_size_kb": os.path.getsize(self.db_path) // 1024 if os.path.exists(self.db_path) else 0,
        }

    def print_trend_report(self, days: int = 7):
        """打印趋势报告到控制台"""
        trend = self.get_trend_data(days)
        if trend["trend"] == "insufficient_data":
            print("  [TREND] 数据不足, 无法生成趋势报告")
            return

        print(f"\n{'='*55}")
        print(f"  活动趋势报告 (近 {days} 天)")
        print(f"{'='*55}")
        print(f"  趋势: {self._trend_label(trend['trend'])}")

        if trend["dates"]:
            print(f"\n  {'日期':<12} {'日均风险':>8} {'峰值风险':>8} {'告警':>8}")
            print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8}")
            for i, d in enumerate(trend["dates"]):
                total_alerts = (trend["alert_counts"]["blue"][i] +
                               trend["alert_counts"]["yellow"][i] +
                               trend["alert_counts"]["orange"][i] +
                               trend["alert_counts"]["red"][i])
                print(f"  {d:<12} {trend['avg_risk'][i]:>8.1f} {trend['max_risk'][i]:>8.1f} {total_alerts:>8}")

        # 恶化告警
        warn = self.get_risk_deterioration_alert(days)
        if warn:
            print(f"\n  {warn}")

        stats = self.get_stats()
        print(f"\n  总计: {stats['total_frames']} 帧, {stats['total_sessions']} 次会话, "
              f"DB {stats['db_size_kb']}KB")
        print(f"{'='*55}\n")

    @staticmethod
    def _trend_label(trend: str) -> str:
        return {"stable": "✅ 平稳", "improving": "📉 改善中",
                "worsening": "⚠️ 恶化中", "insufficient_data": "❓ 数据不足"}.get(trend, trend)


# ════════════════════════════════════════════════════════
# CLI 入口
# ════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Activity Logger — 活动日志查询与分析")
    ap.add_argument("--db", default=r"E:\老人跌倒\logs\activity.db",
                    help="数据库路径 (默认 E:\\老人跌倒\\logs\\activity.db)")
    ap.add_argument("--log-dir", default=r"E:\老人跌倒\logs",
                    help="日志目录 (用于 init)")
    ap.add_argument("action", nargs="?", default="trend",
                    choices=["trend", "stats", "alerts", "segments", "init"],
                    help="操作: trend/stats/alerts/segments/init")
    ap.add_argument("--days", type=int, default=7, help="趋势分析天数")
    ap.add_argument("--hours", type=int, default=24, help="查询小时数")
    ap.add_argument("--min-risk", type=float, default=40.0, help="最低风险阈值")
    ap.add_argument("--top", type=int, default=10, help="返回片段数")
    args = ap.parse_args()

    logger = ActivityLogger(args.log_dir)

    if args.action == "init":
        print(f"✅ 数据库已初始化: {logger.db_path}")
        return

    if args.action == "trend":
        logger.print_trend_report(days=args.days)

    elif args.action == "stats":
        stats = logger.get_stats()
        print(f"总帧数: {stats['total_frames']}")
        print(f"总会话: {stats['total_sessions']}")
        print(f"24h 告警: {stats['alerts_24h']}")
        print(f"DB 大小: {stats['db_size_kb']}KB")
        if stats['last_log']:
            print(f"最后记录: {stats['last_log']['time']} "
                  f"(risk={stats['last_log']['risk']:.0f}, "
                  f"level={stats['last_log']['alert']})")

    elif args.action == "alerts":
        alerts = logger.query_recent_alerts(hours=args.hours, limit=50)
        print(f"\n近 {args.hours}h 告警 ({len(alerts)} 条):")
        for a in alerts:
            print(f"  {a['iso_time']} | {a['alert_type']:>12} | "
                  f"risk={a['risk_score']:.0f} | dir={a['fall_direction']}")

    elif args.action == "segments":
        segments = logger.query_high_risk_segments(
            hours=args.hours, min_risk=args.min_risk, top_n=args.top
        )
        print(f"\n近 {args.hours}h 高风险片段 (min_risk≥{args.min_risk}):")
        for s in segments:
            print(f"  #{s.segment_id} | {s.start_time[:19]} ~ {s.end_time[:19]} | "
                  f"duration={s.duration_sec:.1f}s | "
                  f"peak={s.peak_risk:.0f} avg={s.avg_risk:.0f} | "
                  f"level={s.alert_level} | direction={s.fall_direction}")


if __name__ == "__main__":
    main()
