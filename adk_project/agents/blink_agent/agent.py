# adk_project/agents/blink_agent/agent.py
# Blink Agent (conservative streaming): uses MediaPipe Face Mesh to estimate eye aspect ratio (EAR)
# Finds blinks and flags unusually long no-blink gaps as "suspicious" spans, with tracking gates.

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np
import mediapipe as mp
import cv2
import math

from adk_project.tools.media_io import load_video_frames, stream_video_frames

# ---------------- Tunables (conservative defaults) ----------------
REQ_TRACKED_FRAC     = 0.70   # need >=70% frames with a tracked face
REQ_EYE_WIDTH_PX     = 16.0   # need median eye-corner distance >= 16 px (in original scale)
NO_BLINK_MIN_VIDEO_S = 14.0   # never accuse "no blinks" if clip shorter than this
NO_BLINK_GAP_S       = 14.0   # gap needed to flag; was 12
EAR_THRESH_ABS       = 0.19   # absolute EAR close threshold
MIN_CLOSED_FRAMES    = 3      # frames under EAR threshold to consider a blink

# ---------------- EAR helpers ----------------
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def _ear_from_pts(pts: np.ndarray) -> float:
    """Eye Aspect Ratio from 6 points (mediapipe indices).
       pts order: [outer, upper1, upper2, inner, lower2, lower1]"""
    v = (np.linalg.norm(pts[1] - pts[4]) + np.linalg.norm(pts[2] - pts[5])) / 2.0
    h = np.linalg.norm(pts[0] - pts[3]) + 1e-6
    return float(v / h)

# ---------------- Streaming blink detection ----------------
def detect_blinks_stream(
    frame_gen,
    fps: int = 25,
    ear_thresh: float = EAR_THRESH_ABS,
    min_closed_frames: int = MIN_CLOSED_FRAMES,
    proc_width: Optional[int] = 320,
) -> Tuple[List[float], List[float], List[float], Dict]:
    """Streaming blink detector: optimized for speed and memory."""
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
    last_ts: float = 0.0

    eye_w_samples: List[float] = []

    def _eye_width(px_pts: np.ndarray) -> float:
        return float(np.linalg.norm(px_pts[0] - px_pts[3]))

    try:
        for frame, ts in frame_gen:
            total_frames += 1
            last_ts = float(ts)
            h, w = frame.shape[:2]

            # Resize for speed but keep scale factor so eye width is in original-px scale
            if proc_width is not None and w > proc_width:
                scale = proc_width / w
                proc_frame = cv2.resize(frame, (proc_width, int(h * scale)), interpolation=cv2.INTER_AREA)
                proc_h, proc_w = proc_frame.shape[:2]
            else:
                proc_frame = frame
                scale = 1.0
                proc_h, proc_w = h, w

            # MediaPipe expects RGB; MoviePy frames are RGB already. If your source is BGR, uncomment:
            # proc_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(proc_frame)

            if not results.multi_face_landmarks:
                ear_series.append(np.nan)
                times.append(last_ts)
                missing_count += 1
                closed_streak = 0
                closed_start_ts = None
                continue

            face = results.multi_face_landmarks[0]
            # Convert normalized coords to ORIGINAL scale pixels (divide by scale)
            lm = np.array([[lmk.x * proc_w / scale, lmk.y * proc_h / scale] for lmk in face.landmark],
                          dtype=np.float32)

            L = lm[LEFT_EYE]
            R = lm[RIGHT_EYE]
            ear_left = _ear_from_pts(L)
            ear_right = _ear_from_pts(R)
            ear = float((ear_left + ear_right) / 2.0)

            ear_series.append(ear)
            times.append(last_ts)

            # Face quality proxy
            eye_w = (_eye_width(L) + _eye_width(R)) / 2.0
            eye_w_samples.append(eye_w)

            # Blink detection (absolute EAR)
            if ear < ear_thresh:
                if closed_streak == 0:
                    closed_start_ts = last_ts
                closed_streak += 1
            else:
                if closed_streak >= min_closed_frames and closed_start_ts is not None:
                    center_ts = closed_start_ts + (last_ts - closed_start_ts) / 2.0
                    blink_times.append(center_ts)
                    blink_durations.append(last_ts - closed_start_ts)
                closed_streak = 0
                closed_start_ts = None
    finally:
        face_mesh.close()

    # Tail case
    if closed_streak >= min_closed_frames and closed_start_ts is not None:
        center_ts = closed_start_ts + (last_ts - closed_start_ts) / 2.0
        blink_times.append(center_ts)
        blink_durations.append(last_ts - closed_start_ts)

    # Smooth EAR (3-tap)
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
    metrics: Dict = {
        "frame_count": int(total_frames),
        "missing_frames": int(missing_count),
        "nan_fraction": float(missing_count / total_frames) if total_frames else 1.0,
        "blink_count": int(len(blink_times)),
        "avg_blink_duration_s": float(np.mean(blink_durations)) if blink_durations else 0.0,
        "blink_durations_s": [float(x) for x in blink_durations],
        "ear_thresh_used": float(ear_thresh),
        "mode": "absolute",
    }

    tracked_fraction = 1.0 - metrics["nan_fraction"]
    metrics["tracked_fraction"] = float(tracked_fraction)
    if eye_w_samples:
        metrics["eye_width_px_median"] = float(np.median(eye_w_samples))

    # Relative-drop recovery ONLY IF absolute found zero blinks but tracking is decent
    def _rederive_blinks_rel(series: np.ndarray, ts_list: List[float], min_closed: int):
        if series.size == 0 or len(ts_list) != series.size:
            return [], []
        base = float(np.nanmedian(series))
        delta = 0.12  # conservative drop below median
        thr = base - delta
        bt, bd = [], []
        streak = 0
        start_ts = None
        for i, v in enumerate(series):
            if v < thr:
                if streak == 0:
                    start_ts = ts_list[i]
                streak += 1
            else:
                if streak >= min_closed and start_ts is not None:
                    end_ts = ts_list[i]
                    center = start_ts + (end_ts - start_ts)/2.0
                    bt.append(float(center)); bd.append(float(end_ts - start_ts))
                streak = 0; start_ts = None
        if streak >= min_closed and start_ts is not None:
            end_ts = ts_list[-1]
            center = start_ts + (end_ts - start_ts)/2.0
            bt.append(float(center)); bd.append(float(end_ts - start_ts))
        return bt, bd

    if metrics["blink_count"] == 0 and tracked_fraction >= 0.60 and len(ear_sm_list) >= max(2*fps, 30):
        bt2, bd2 = _rederive_blinks_rel(np.array(ear_sm_list, dtype=np.float32), times, min_closed_frames)
        if len(bt2) > 0:
            blink_times, blink_durations = bt2, bd2
            metrics["blink_count"] = len(bt2)
            metrics["blink_durations_s"] = bd2
            metrics["ear_thresh_used"] = float('nan')
            metrics["mode"] = "relative_drop"

    return blink_times, ear_sm_list, blink_durations, metrics

# ---------------- Suspicious blink spans ----------------
def _no_blink_spans(blink_times: List[float], video_duration_s: float,
                    max_gap_ok: float = NO_BLINK_GAP_S) -> List[Tuple[float,float,str]]:
    """Build (start,end,reason) spans for long no-blink gaps."""
    spans: List[Tuple[float,float,str]] = []
    if not blink_times:
        spans.append((0.0, float(video_duration_s), f"No blinks detected for {video_duration_s:.1f}s"))
        return spans
    times = [0.0] + sorted(blink_times) + [float(video_duration_s)]
    for a, b in zip(times[:-1], times[1:]):
        gap = float(b - a)
        if gap >= max_gap_ok:
            spans.append((float(a), float(b), f"No blink for {gap:.1f}s (≥ {max_gap_ok:.0f}s)"))
    return spans

# ---------------- Top-level analysis ----------------
@dataclass
class BlinkResult:
    suspicious_spans: List[dict]
    metric: str = "blink_no_gap"
    details: Dict = None

def analyze_blinks(video_path: str,
                   fps: int = 25,
                   ear_thresh: float = EAR_THRESH_ABS,
                   min_closed_frames: int = MIN_CLOSED_FRAMES,
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

    tracked = float(1.0 - metrics.get("nan_fraction", 1.0))
    eye_w_med = float(metrics.get("eye_width_px_median", 0.0))

    details: Dict = {
        "blink_times_s": [float(x) for x in (blink_times or [])],
        "blink_durations_s": [float(x) for x in (blink_durations or [])],
        "video_duration_s": float(vid_dur),
        "fps": int(fps),
        "metrics": metrics,
        "coverage": float(round(tracked, 3)),
        "eye_width_px_median": float(round(eye_w_med, 1)),
        "gates": {
            "req_tracked_frac": REQ_TRACKED_FRAC,
            "req_eye_width_px": REQ_EYE_WIDTH_PX,
            "no_blink_min_video_s": NO_BLINK_MIN_VIDEO_S,
            "no_blink_gap_s": NO_BLINK_GAP_S,
        }
    }

    # ------------- GATES (fail → return no spans) -------------
    # 1) poor tracking
    if tracked < REQ_TRACKED_FRAC:
        details["description"] = f"insufficient eye tracking (tracked={tracked:.2f}); skip blink judgement"
        return BlinkResult(suspicious_spans=[], details=details)

    # 2) face too small
    if eye_w_med < REQ_EYE_WIDTH_PX:
        details["description"] = f"face too small (eye_width_med={eye_w_med:.1f}px); skip blink judgement"
        return BlinkResult(suspicious_spans=[], details=details)

    # 3) clip too short to accuse "no blinks"
    if float(vid_dur) < NO_BLINK_MIN_VIDEO_S:
        details["description"] = f"video too short for no-blink judgement ({vid_dur:.1f}s)"
        return BlinkResult(suspicious_spans=[], details=details)

    # ------------- Build spans -------------
    spans_triplets = _no_blink_spans(blink_times, float(vid_dur), max_gap_ok=NO_BLINK_GAP_S)

    # Keep only long gaps ≥ NO_BLINK_GAP_S (already enforced), convert to dicts
    out_spans = [{"start": float(s), "end": float(e), "reason": str(r)} for (s, e, r) in spans_triplets]

    # Friendly description
    if metrics.get("blink_count", 0) == 0:
        details["description"] = f"0 blinks in {vid_dur:.1f}s; gating passed (tracked {tracked:.0%})"
    else:
        details["description"] = f"{metrics['blink_count']} blinks; avg dur {metrics['avg_blink_duration_s']:.2f}s"

    return BlinkResult(suspicious_spans=out_spans, details=details)

# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Blink Agent (conservative streaming)")
    p.add_argument("--in", dest="inp", required=True, help="Path to video (mp4)")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    p.add_argument("--stream", action="store_true", help="Stream frames (don’t load all frames into memory)")
    p.add_argument("--ear-thresh", type=float, default=EAR_THRESH_ABS)
    p.add_argument("--min-closed-frames", type=int, default=MIN_CLOSED_FRAMES)
    p.add_argument("--proc-width", type=int, default=320)
    p.add_argument("--json", action="store_true", help="Print JSON only")
    args = p.parse_args()

    res = analyze_blinks(
        args.inp,
        fps=args.fps,
        ear_thresh=args.ear_thresh,
        min_closed_frames=args.min_closed_frames,
        proc_width=args.proc_width,
        frame_stride=args.frame_stride,
        use_stream=args.stream or args.frame_stride > 1,
    )

    payload = {
        "agent": "blink",
        "metric": "blink_no_gap",
        "suspicious_spans": res.suspicious_spans,
        "details": res.details,
    }
    print(json.dumps(payload) if args.json else json.dumps(payload, indent=2))


