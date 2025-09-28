# adk_project/agents/blink_agent/agent.py
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import mediapipe as mp
import cv2

from adk_project.tools.media_io import load_video_frames, stream_video_frames

# ---------------- EAR helpers ----------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def _ear_from_landmarks(landmarks: np.ndarray, eye_idx: List[int]) -> float:
    """Compute Eye Aspect Ratio (EAR) for a single eye"""
    pts = landmarks[eye_idx]
    v = (np.linalg.norm(pts[1] - pts[4]) + np.linalg.norm(pts[2] - pts[5])) / 2.0
    h = np.linalg.norm(pts[0] - pts[3]) + 1e-6
    return v / h

# ---------------- Streaming blink detection ----------------
def detect_blinks_stream(
    frame_gen,
    fps: int = 25,
    ear_thresh: float = 0.19,
    min_closed_frames: int = 3,
    proc_width: Optional[int] = 320,
):
    """Streaming blink detector: fully online and memory-efficient"""

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    blink_times: List[float] = []
    blink_durations: List[float] = []
    ear_series: List[float] = []
    times: List[float] = []

    total_frames = 0
    missing_count = 0
    closed_streak = 0
    closed_start_ts: Optional[float] = None

    try:
        for frame, ts in frame_gen:
            total_frames += 1
            h, w = frame.shape[:2]

            # Resize for faster processing
            if proc_width is not None and w > proc_width:
                scale = proc_width / w
                proc_frame = cv2.resize(frame, (proc_width, int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                proc_frame = frame
                scale = 1.0

            results = face_mesh.process(proc_frame)

            if not results.multi_face_landmarks:
                ear_series.append(np.nan)
                times.append(ts)
                missing_count += 1
                closed_streak = 0
                closed_start_ts = None
                continue

            # Map landmarks to original frame coordinates
            face = results.multi_face_landmarks[0]
            pts = np.array([(lm.x * proc_frame.shape[1] / scale, lm.y * proc_frame.shape[0] / scale)
                            for lm in face.landmark], dtype=np.float32)

            # Compute EAR
            le = _ear_from_landmarks(pts, LEFT_EYE)
            re = _ear_from_landmarks(pts, RIGHT_EYE)
            ear = (le + re) / 2.0
            ear_series.append(ear)
            times.append(ts)

            # Online blink detection
            if ear < ear_thresh:
                if closed_streak == 0:
                    closed_start_ts = ts
                closed_streak += 1
            else:
                if closed_streak >= min_closed_frames and closed_start_ts is not None:
                    center_ts = closed_start_ts + (ts - closed_start_ts) / 2.0
                    blink_times.append(center_ts)
                    blink_durations.append(max(0.0, ts - closed_start_ts))
                closed_streak = 0
                closed_start_ts = None
    finally:
        face_mesh.close()

    # Tail case: last frames closed
    if closed_streak >= min_closed_frames and closed_start_ts is not None:
        center_ts = closed_start_ts + (ts - closed_start_ts) / 2.0
        blink_times.append(center_ts)
        blink_durations.append(max(0.0, ts - closed_start_ts))

    # Smooth EAR series
    ear_arr = np.array(ear_series, dtype=np.float32)
    if ear_arr.size > 0:
        nan_mask = np.isnan(ear_arr)
        if np.any(~nan_mask):
            idxs = np.arange(ear_arr.size)
            ear_arr[nan_mask] = np.interp(idxs[nan_mask], idxs[~nan_mask], ear_arr[~nan_mask])
        ear_sm_list = np.convolve(ear_arr, np.ones(3)/3.0, mode='same').tolist()
    else:
        ear_sm_list = []

    # Metrics
    metrics = {
        "frame_count": total_frames,
        "missing_frames": missing_count,
        "nan_fraction": missing_count / total_frames if total_frames > 0 else 1.0,
        "blink_count": len(blink_times),
        "avg_blink_duration_s": float(np.mean(blink_durations)) if blink_durations else 0.0,
        "blink_durations_s": blink_durations,
    }

    if len(blink_times) >= 2:
        intervals = list(np.diff(sorted(blink_times)))
        metrics["blink_intervals_s"] = intervals
        metrics["interval_cov"] = float(np.std(intervals)/np.mean(intervals)) if np.mean(intervals) > 0 else float('inf')
    else:
        metrics["blink_intervals_s"] = []
        metrics["interval_cov"] = None

    return blink_times, ear_sm_list, blink_durations, metrics

# ---------------- Suspicious blink spans ----------------
def suspicious_no_blink_spans(blink_times: List[float], video_duration_s: float, max_gap_ok: float = 12.0):
    spans = []
    if not blink_times:
        spans.append((0.0, video_duration_s, f"No blinks detected for {video_duration_s:.1f}s"))
        return spans
    times = [0.0] + sorted(blink_times) + [video_duration_s]
    for a, b in zip(times[:-1], times[1:]):
        gap = b - a
        if gap >= max_gap_ok:
            spans.append((a, b, f"No blink for {gap:.1f}s (>= {max_gap_ok:.0f}s)"))
    return spans

# ---------------- Robotic blink detection ----------------
def detect_robotic_blinks_from_metrics(metrics: Dict, min_count: int = 4, cov_thresh: float = 0.2) -> Optional[Dict]:
    intervals = metrics.get("blink_intervals_s", [])
    if len(intervals) < min_count:
        return None
    mean = float(np.mean(intervals))
    std = float(np.std(intervals))
    cov = std / mean if mean > 0 else float('inf')
    if cov <= cov_thresh:
        return {"mean_interval_s": mean, "std_interval_s": std, "cov": cov,
                "reason": f"Blink intervals too uniform (CoV={cov:.2f}, mean={mean:.2f}s)"}
    return None

# ---------------- Top-level analysis ----------------
@dataclass
class BlinkResult:
    suspicious_spans: List[dict]
    metric: str = "blink_no_gap"
    details: Dict = None
    score: float = 0.0
    classification: str = "unknown"

def analyze_blinks(video_path: str,
                   fps: int = 25,
                   ear_thresh: float = 0.19,
                   min_closed_frames: int = 3,
                   proc_width: Optional[int] = 320,
                   frame_stride: int = 1,
                   use_stream: bool = False) -> BlinkResult:
    """Analyze blinks in a video (streaming preferred)."""
    if use_stream:
        frame_gen, vid_dur = stream_video_frames(video_path, fps=fps, frame_stride=frame_stride)
    else:
        frames, times_s, vid_dur = load_video_frames(video_path, fps=fps, frame_stride=frame_stride)
        frame_gen = ((frames[i], times_s[i]) for i in range(len(frames)))

    blink_times, ear_series, blink_durations, metrics = detect_blinks_stream(
        frame_gen, fps=fps, ear_thresh=ear_thresh, min_closed_frames=min_closed_frames, proc_width=proc_width
    )

    spans = suspicious_no_blink_spans(blink_times, vid_dur)
    robotic = detect_robotic_blinks_from_metrics(metrics)
    if robotic:
        spans.append((0.0, float(vid_dur), robotic["reason"]))

    details = {
        "blink_times_s": blink_times,
        "blink_durations_s": blink_durations,
        "video_duration_s": float(vid_dur),
        "fps": int(fps),
        "metrics": metrics,
    }

    if metrics["blink_count"] == 0:
        details["description"] = f"0 blinks in {vid_dur:.1f}s video"
    elif robotic is not None:
        details["description"] = robotic["reason"]
    else:
        details["description"] = f"{metrics['blink_count']} blinks, avg dur {metrics['avg_blink_duration_s']:.2f}s"
